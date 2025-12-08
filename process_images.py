import os
import cv2
import numpy as np
import time
from collections import deque
from paddlex import create_pipeline

############################
# 1. 参数与配置
############################

# 说明：多数 PaddleX/Cityscapes 风格推理输出为 trainId：
#   road=0, sidewalk=1, person=11（注意：11 通常不是地面！）
# 如为 labelId：road=7, sidewalk=8。请用 DEBUG_UNIQUE_IDS 检查后再调整。
GROUND_IDS = [0, 1]  # 默认按 trainId：road, sidewalk

# 决策规则参数
TH_GROUND = 0.15        # 地面比例阈值：大于这个值认为“这一块有明显地面”
DELTA_SIDE = 0.10       # 左右地面比例差值，大于此认定为“多出一侧通路”

# 判定稳定事件所需的帧数（这里只处理单张图片，可设为 1）
STABLE_FRAMES = 1

# 输入输出文件夹
DATA_DIR = "data"   # 输入图片目录
RES_DIR = "res"     # 输出结果目录

# 调试配置
DEBUG_UNIQUE_IDS = True   # 首张图打印 pred 的唯一类别 id，帮助确认 GROUND_IDS
_printed_unique_once = False

############################
# 2. PaddleX 模型加载
############################

def load_seg_model(model_dir):
    """
    加载 PaddleX 导出的语义分割推理模型.
    model_dir: 推理模型目录，例如 'inference_model'
    说明：部分 PaddleX 版本通过 create_pipeline 使用默认配置/环境加载，
          如需显式加载本地导出模型，请在此处接入对应 API。
    """
    model = create_pipeline(pipeline="semantic_segmentation")
    return model

def get_ground_mask(pred):
    """
    pred: H x W 的类别 id 图 (int)
    返回: H x W 的 bool 数组，True 表示地面 (road/sidewalk)
    """
    ground_mask = np.isin(pred, GROUND_IDS)
    return ground_mask

############################
# 3. 分割结果后处理（增强版）
############################

def refine_ground_mask(mask, use_horizon=True, horizon_ratio=0.4, min_area_ratio=0.002):
    """
    通过更稳健的流程优化分割出的地面掩码：
    - 位置先验：可选，清除上方（天空/建筑）区域
    - 连通域去小区域：删除小于全图一定比例的噪点
    - 自适应核开运算：去细碎噪声
    - 自适应核闭运算：小幅填洞，避免过度连片
    - 轻度中值滤波：平滑边缘
    参数：
      use_horizon: 是否使用地平线先验（建议 True）
      horizon_ratio: 清除上方比例（例如 0.4 表示清除上方 40%）
      min_area_ratio: 小连通域面积阈值占比（0.001~0.005 常用）
    返回：bool 掩码
    """
    h, w = mask.shape
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # 1) 位置先验：上方通常为天空/建筑，易误检为地面
    if use_horizon and h > 0:
        horizon = int(h * horizon_ratio)
        horizon = np.clip(horizon, 0, h)
        mask_uint8[:horizon, :] = 0

    # 2) 去小连通域
    if np.count_nonzero(mask_uint8) > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        min_area = max(1, int(h * w * float(min_area_ratio)))
        keep = np.zeros_like(mask_uint8)
        for i in range(1, num_labels):  # 跳过背景 0
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
    else:
        keep = mask_uint8

    # 3) 自适应核大小（随分辨率变化），取奇数
    k = max(3, int(min(h, w) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 4) 开运算去细噪
    opened = cv2.morphologyEx(keep, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) 闭运算小幅填洞
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) 轻度中值滤波
    smoothed = cv2.medianBlur(closed, 3)

    return smoothed > 0

############################
# 4. 根据 ground_mask 计算决策点事件（单张图）
############################

def detect_outdoor_events(ground_mask,
                          th_ground=TH_GROUND,
                          delta_side=DELTA_SIDE):
    """
    输入: ground_mask (H x W, bool)
    输出: events: list of (type, direction)
       type 可为: "TURN", "CROSS", "T_JUNCTION", "OBSTACLE_CENTER_BYPASSABLE"
       direction: "LEFT", "RIGHT", "AHEAD" 或 None
    """
    events = []

    h, w = ground_mask.shape

    # 下半部分代表视障者脚前区域
    roi = ground_mask[int(h * 0.5):, :]  # 可根据实际情况调整 0.5 的比例

    # 左中右三块
    left = roi[:, :w // 3]
    center = roi[:, w // 3: 2 * w // 3]
    right = roi[:, 2 * w // 3:]

    def ratio(region):
        # region 是 bool 数组, True 表示地面
        return float(np.mean(region))

    gl = ratio(left)
    gc = ratio(center)
    gr = ratio(right)

    straight = gc > th_ground

    # 1. 右侧出现明显新通路(右侧地面多于左侧且明显大)
    if straight and (gr - gl) > delta_side and gr > th_ground:
        events.append(("TURN", "RIGHT"))

    # 2. 左侧出现明显新通路
    if straight and (gl - gr) > delta_side and gl > th_ground:
        events.append(("TURN", "LEFT"))

    # 3. 十字路口/大路口：三块都有不少地面
    if gc > th_ground and gl > th_ground and gr > th_ground:
        events.append(("CROSS", "AHEAD"))

    # 4. 丁字路口：前方地面少，但两侧多
    if gc < th_ground and gl > th_ground and gr > th_ground:
        events.append(("T_JUNCTION", "STOP_AHEAD"))

    # 5. 中央有障碍，但左右至少一侧可绕行
    if gc < th_ground and (gl > th_ground or gr > th_ground):
        events.append(("OBSTACLE_CENTER_BYPASSABLE", None))

    return events

############################
# 5. 事件转成字符串（用于文件名）
############################

def event_to_tag(event):
    """
    event: (type, direction)
    输出: 例如 'TURN_LEFT', 'CROSS', 'T_JUNCTION', 'OBSTACLE_CENTER_BYPASSABLE'
    """
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
    """
    将地面区域以半透明绿色覆盖到原图上，帮助可视化。
    """
    vis = image.copy()
    green = np.zeros_like(vis, dtype=np.uint8)
    green[:] = (0, 255, 0)
    mask_3c = np.stack([ground_mask] * 3, axis=-1)

    vis[mask_3c] = cv2.addWeighted(
        vis, 1 - alpha, green, alpha, 0
    )[mask_3c]
    return vis

############################
# 6. 工具函数：确保 pred 与原图同尺寸
############################

def ensure_pred_size(pred, target_shape_hw):
    """
    若分割输出尺寸与原图不同，使用最近邻缩放到原图大小。
    pred: HxW ndarray（int）
    target_shape_hw: (H, W)
    """
    th, tw = target_shape_hw
    ph, pw = pred.shape[:2]
    if (ph, pw) != (th, tw):
        pred_resized = cv2.resize(pred.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
        return pred_resized.astype(pred.dtype)
    return pred

############################
# 7. 针对单张图片处理：推理 + 决策点 + 保存
############################

def process_single_image(model, img_path, save_dir):
    """
    对单张图片进行：
    1) 读取 & 分割
    2) 结果后处理
    3) 决策点识别
    4) 若有决策点，则保存到 save_dir，文件名加上决策点类型
    """
    global _printed_unique_once

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读取失败: {img_path}")
        return

    output = model.predict(input=img_path, target_size=-1)
    pred = None
    for res in output:
        pred = res.json
    pred = np.array(pred['res']['pred'][0])

    if not isinstance(pred, np.ndarray) or pred.ndim != 2:
        print(f"[ERROR] 预测结果 pred 不是 HxW 的 np.ndarray，请检查 model.predict 的返回格式.")
        return

    # 保证与原图一致大小
    pred = ensure_pred_size(pred, img.shape[:2])

    # 调试：打印一次唯一类别 id，帮助确认 GROUND_IDS 是否需要调整
    if DEBUG_UNIQUE_IDS and not _printed_unique_once:
        uniq = np.unique(pred)
        print(f"[DEBUG] {os.path.basename(img_path)} unique class ids: {uniq}")
        print(f"[DEBUG] 当前 GROUND_IDS = {GROUND_IDS} （如果不匹配，请根据模型映射修改）")
        _printed_unique_once = True

    # ---------- 2. 获取原始地面 mask ----------
    ground_mask = get_ground_mask(pred)

    # ---------- 3. 后处理，优化 mask ----------
    refined_mask = refine_ground_mask(ground_mask, use_horizon=True, horizon_ratio=0.4, min_area_ratio=0.002)

    # ---------- 4. 基于优化后的 mask 检测决策点 ----------
    events = detect_outdoor_events(refined_mask)

    if not events:
        # 没有任何决策点，就不保存（根据你需求可改为仍然保存）
        print(f"[INFO] {os.path.basename(img_path)} 未识别到决策点，跳过保存")
        return

    # ---------- 5. 叠加可视化（使用优化后的 mask） ----------
    vis = overlay_ground_mask(img, refined_mask)

    # 在图像上写上事件文字方便调试
    y0 = 30
    for event in events:
        tag = event_to_tag(event)
        cv2.putText(vis, tag, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)
        y0 += 40

    # ---------- 6. 拼接输出文件名 ----------
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)

    # 可能出现多个事件类型，将它们的 tag 用 '_' 连接起来
    tags = [event_to_tag(e) for e in events]
    tags_str = "_".join(sorted(set(tags)))  # 去重并排序一下，避免重复

    out_name = f"{name}_{tags_str}{ext}"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)

    # ---------- 7. 保存结果图 ----------
    cv2.imwrite(out_path, vis)
    print(f"[SAVE] {out_path}")


def main():
    model_dir = "inference_model"  # TODO: 如需显式加载本地导出的推理模型，请在 load_seg_model 中接入
    model = load_seg_model(model_dir)

    os.makedirs(RES_DIR, exist_ok=True)

    # 遍历 data 目录下的所有图片文件
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        # 简单筛选一下常见图片后缀
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        print(f"[PROCESS] {fpath}")
        process_single_image(model, fpath, RES_DIR)


if __name__ == "__main__":
    main()
