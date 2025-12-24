import os
import cv2
import numpy as np
import argparse
import json

# PaddleSeg
import paddle
from paddleseg.core import predict as seg_predict
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose

############################
# 1. 参数与配置
############################

# Cityscapes 中可行走地面的类别 id
GROUND_IDS = [0, 1]  # 0: road, 1: sidewalk

# 决策规则参数
TH_GROUND = 0.15        # 地面比例阈值：大于这个值认为“这一块有明显地面”
DELTA_SIDE = 0.10       # 左右地面比例差值，大于此认定为“多出一侧通路”
HOLE_MAX_AREA_RATIO = 0.0015   # 填充地面内的小空洞(如树叶)的最大面积占整幅图比例
OBS_MIN_AREA_RATIO = 0.01      # 认为是“脚下中央障碍”所需的最小面积(相对ROI面积)
OBS_MIN_WIDTH_RATIO = 0.06     # 认为是“脚下中央障碍”所需的最小水平宽度(相对整图宽度)

DATA_DIR = "data"
RES_DIR = "res"


############################
# 2. PaddleSeg OCRNet 加载
############################

def load_ocrnet(model_config: str, model_weights: str, device: str = "gpu"):
    """加载 PaddleSeg OCRNet（或任意 PaddleSeg config+weights）。

    参数
    - model_config: PaddleSeg 的配置文件路径（.yml）
    - model_weights: 对应的权重文件路径（.pdparams）
    - device: "gpu" / "cpu"，也可用 "gpu:0" 形式

    返回
    - model: paddle.nn.Layer
    - transforms: paddleseg.transforms.Compose
    """
    # device
    if device.startswith("gpu"):
        paddle.set_device(device)
    elif device.startswith("cpu"):
        paddle.set_device("cpu")
    else:
        # 兜底
        paddle.set_device(device)

    # 读取 config
    from paddleseg.cvlibs import Config
    cfg = Config(model_config)

    # 构建 transforms
    transforms = Compose(cfg.val_transforms)

    # 构建模型
    model = cfg.model
    model.eval()

    # 加载权重
    state_dict = paddle.load(model_weights)
    model.set_state_dict(state_dict)

    print(f"[INFO] 已加载 PaddleSeg 模型: {model_config}")
    print(f"[INFO] 权重: {model_weights}")
    print(f"[INFO] device: {paddle.get_device()}")

    return model, transforms


############################
# 3. 业务逻辑（与原脚本一致）
############################

def get_ground_mask(pred: np.ndarray) -> np.ndarray:
    return np.isin(pred, GROUND_IDS)


def simple_refine_ground_mask(ground_mask_bool,
                              kernel_rel=0.008,
                              min_area_ratio=0.0005,
                              keep_only_bottom_connected=False,
                              hole_max_area_ratio=0.0):
    H, W = ground_mask_bool.shape
    b = (ground_mask_bool.astype(np.uint8) * 255)

    k = max(3, int(min(H, W) * float(kernel_rel)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((b > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (b > 0)

    area_th = H * W * float(min_area_ratio)
    keep = np.zeros((H, W), dtype=np.uint8)

    def touches_bottom(lbl_id):
        return np.any(labels[-1, :] == lbl_id) or np.any(labels[-2, :] == lbl_id)

    kept_ids = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= area_th:
            if not keep_only_bottom_connected or touches_bottom(i):
                keep[labels == i] = 255
                kept_ids.append(i)

    if keep_only_bottom_connected and len(kept_ids) == 0:
        i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        keep[labels == i] = 255

    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=1)

    if hole_max_area_ratio and hole_max_area_ratio > 0:
        mask_bin = (keep > 0).astype(np.uint8)
        inv = (255 - mask_bin * 255).copy()
        flood_mask = np.zeros((H + 2, W + 2), np.uint8)
        cv2.floodFill(inv, flood_mask, (0, 0), 0)
        holes = (inv == 255)

        num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        hole_area_th = H * W * float(hole_max_area_ratio)
        for i_h in range(1, num_h):
            if stats_h[i_h, cv2.CC_STAT_AREA] <= hole_area_th:
                keep[labels_h == i_h] = 255

    return (keep > 0)


def suppress_small_roi_obstacles(ground_mask,
                                 obs_min_area_ratio=0.01,
                                 obs_min_width_ratio=0.06,
                                 roi_top_rel=0.5):
    H, W = ground_mask.shape
    top = int(H * float(roi_top_rel))
    roi = ground_mask[top:, :].copy()

    obs = (~roi).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(obs, connectivity=8)
    if num <= 1:
        return ground_mask

    roi_area = roi.shape[0] * roi.shape[1]
    area_th = roi_area * float(obs_min_area_ratio)
    width_th = W * float(obs_min_width_ratio)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        if area < area_th or width < width_th:
            roi[labels == i] = True

    out = ground_mask.copy()
    out[top:, :] = roi
    return out


def detect_outdoor_events(ground_mask,
                          th_ground=TH_GROUND,
                          delta_side=DELTA_SIDE):
    events = []

    h, w = ground_mask.shape
    roi = ground_mask[int(h * 0.5):, :]

    left = roi[:, :w // 3]
    center = roi[:, w // 3: 2 * w // 3]
    right = roi[:, 2 * w // 3:]

    def ratio(region):
        return float(np.mean(region))

    gl = ratio(left)
    gc = ratio(center)
    gr = ratio(right)

    straight = gc > th_ground

    if straight and (gr - gl) > delta_side and gr > th_ground:
        events.append(("TURN", "RIGHT"))
    if straight and (gl - gr) > delta_side and gl > th_ground:
        events.append(("TURN", "LEFT"))
    if gc > th_ground and gl > th_ground and gr > th_ground:
        events.append(("CROSS", "AHEAD"))
    if gc < th_ground and gl > th_ground and gr > th_ground:
        events.append(("T_JUNCTION", "STOP_AHEAD"))
    if gc < th_ground and (gl > th_ground or gr > th_ground):
        events.append(("OBSTACLE_CENTER_BYPASSABLE", None))

    return events


def event_to_tag(event):
    etype, edir = event
    if etype == "TURN":
        if edir == "RIGHT":
            return "TURN_RIGHT"
        elif edir == "LEFT":
            return "TURN_LEFT"
        else:
            return "TURN"
    elif etype == "CROSS":
        return "CROSS"
    elif etype == "T_JUNCTION":
        return "T_JUNCTION"
    elif etype == "OBSTACLE_CENTER_BYPASSABLE":
        return "OBSTACLE_CENTER_BYPASSABLE"
    return "UNKNOWN"


def overlay_ground_mask(image, ground_mask, alpha=0.4):
    vis = image.copy()
    green = np.zeros_like(vis, dtype=np.uint8)
    green[:] = (0, 255, 0)
    mask_3c = np.stack([ground_mask] * 3, axis=-1)

    vis[mask_3c] = cv2.addWeighted(vis, 1 - alpha, green, alpha, 0)[mask_3c]
    return vis


############################
# 4. PaddleSeg 输出解析为 HxW 类别图
############################

def paddleseg_predict_label_map(model, transforms, img_bgr: np.ndarray) -> np.ndarray:
    """对单张 BGR 图像做 PaddleSeg 推理，返回 HxW 的 int label_map。"""

    # PaddleSeg transforms 通常期望 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    data = {"img": img_rgb}
    data = transforms(data)
    im = data["img"]

    # HWC -> CHW
    if im.ndim == 3:
        im = np.transpose(im, (2, 0, 1))

    im = im[np.newaxis, ...].astype("float32")
    im_t = paddle.to_tensor(im)

    with paddle.no_grad():
        logits = model(im_t)
        # 有些模型返回 list/tuple
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pred = paddle.argmax(logits, axis=1)  # N,H,W

    label_map = pred.numpy()[0].astype(np.int32)
    return label_map


############################
# 5. 单张图推理 + 决策点识别 + 保存
############################

def process_single_image(model, transforms, img_path, save_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读取失败: {img_path}")
        return

    pred = paddleseg_predict_label_map(model, transforms, img)

    ground_mask = simple_refine_ground_mask(
        get_ground_mask(pred),
        kernel_rel=0.008,
        min_area_ratio=0.0005,
        keep_only_bottom_connected=True,
        hole_max_area_ratio=HOLE_MAX_AREA_RATIO
    )

    ground_mask_evt = suppress_small_roi_obstacles(
        ground_mask,
        obs_min_area_ratio=OBS_MIN_AREA_RATIO,
        obs_min_width_ratio=OBS_MIN_WIDTH_RATIO
    )

    events = detect_outdoor_events(ground_mask_evt)

    if not events:
        print(f"[INFO] {os.path.basename(img_path)} 未识别到决策点，跳过保存")
        return

    vis = overlay_ground_mask(img, ground_mask_evt)

    y0 = 30
    for event in events:
        tag = event_to_tag(event)
        cv2.putText(vis, tag, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)
        y0 += 40

    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    tags = [event_to_tag(e) for e in events]
    tags_str = "_".join(sorted(set(tags)))

    out_name = f"{name}_{tags_str}{ext}"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)

    cv2.imwrite(out_path, vis)
    print(f"[SAVE] {out_path}")


############################
# 6. CLI
############################

def main():
    parser = argparse.ArgumentParser(description="使用 PaddleSeg(OCRNet) 语义分割并输出带事件标注的可视化结果")

    parser.add_argument("--config", type=str, required=True, help="PaddleSeg 配置文件路径，例如 configs/ocrnet/ocrnet_hrnetw48_cityscapes_1024x512_80k.yml")
    parser.add_argument("--weights", type=str, required=True, help="PaddleSeg 权重路径，例如 ocrnet_hrnetw48_cityscapes_1024x512_80k.pdparams")
    parser.add_argument("--device", type=str, default="gpu", help="推理设备：gpu / cpu / gpu:0")

    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="输入图片目录")
    parser.add_argument("--out_dir", type=str, default=RES_DIR, help="输出结果目录")

    args = parser.parse_args()

    model, transforms = load_ocrnet(args.config, args.weights, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in os.listdir(args.data_dir):
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        print(f"[PROCESS] {fpath}")
        process_single_image(model, transforms, fpath, args.out_dir)


if __name__ == "__main__":
    main()

