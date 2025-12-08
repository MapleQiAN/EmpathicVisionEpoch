import os
import cv2
import numpy as np
import time
from collections import deque
from paddlex import create_pipeline

############################
# 1. 参数与配置
############################

# Cityscapes 中可行走地面的类别 id
GROUND_IDS = [0, 1]  # 0: road, 1: sidewalk

# 决策规则参数（像素比例逻辑）
TH_GROUND = 0.15        # 地面比例阈值：大于这个值认为“这一块有明显地面”
DELTA_SIDE = 0.10       # 左右地面比例差值，大于此认定为“多出一侧通路”

# 判定稳定事件所需的帧数（这里只处理单张图片，可设为 1）
STABLE_FRAMES = 1

# 输入输出文件夹
DATA_DIR = "data"   # 输入图片目录
RES_DIR = "res"     # 输出结果目录

# 形态与轮廓分析相关参数（第三点实现）
MORPH_KERNEL_SIZE = 5           # 形态学核大小
MIN_REGION_AREA_RATIO = 0.02    # 主连通域最小面积占比（相对整幅图像）
NUM_BANDS = 6                   # 垂直方向分层数（在 ROI 内）
DEBUG_GEOM_VIS = False          # 是否在输出图上画几何调试信息


############################
# 2. PaddleX 模型加载
############################

def load_seg_model(model_dir):
    """
    加载 PaddleX 导出的语义分割推理模型.
    model_dir: 推理模型目录，例如 'inference_model'
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
# 3. 形态学清理与轮廓/几何特征（第三点）
############################

def _clean_mask_and_extract_main_region(mask_bool):
    """
    对 ground mask 做形态学开闭与小区域移除，提取最大连通域。
    返回:
      main_mask: np.uint8 的 0/255 掩码（清理后最大地面区域），若无则全 0
      contour: 最大区域的轮廓（None 表示无）
    """
    h, w = mask_bool.shape
    mask = (mask_bool.astype(np.uint8) * 255)

    # 形态学开闭，平滑噪声
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # 找外部轮廓
    cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts_info) == 3:
        _, contours, _ = cnts_info
    else:
        contours, _ = cnts_info

    if not contours:
        return np.zeros_like(mask), None

    # 选面积最大且超过阈值的区域
    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    contour = contours[idx]
    area = areas[idx]

    if area < MIN_REGION_AREA_RATIO * (h * w):
        return np.zeros_like(mask), None

    main_mask = np.zeros_like(mask)
    cv2.drawContours(main_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return main_mask, contour


def _compute_band_stats(main_mask, roi_y0, roi_y1, num_bands=NUM_BANDS):
    """
    在 ROI [roi_y0, roi_y1) 内将 main_mask 分成 num_bands 个水平条带，
    计算每条带的：
      - cover_ratio: 掩码像素占条带像素比例
      - left_cover, center_cover, right_cover: 左中右三分区覆盖率
      - mean_width: 行宽度的平均值（按像素），归一化到 [0,1]（除以整幅图宽度）
      - centroid_x: 掩码像素横坐标均值（归一化到 [-0.5, 0.5]）
    返回：列表，每个元素为 dict。
    """
    h, w = main_mask.shape
    roi_y0 = int(max(0, min(h, roi_y0)))
    roi_y1 = int(max(0, min(h, roi_y1)))
    if roi_y1 <= roi_y0:
        return []

    band_h = max(1, (roi_y1 - roi_y0) // num_bands)

    stats = []
    for i in range(num_bands):
        y0 = roi_y0 + i * band_h
        y1 = roi_y0 + (i + 1) * band_h if i < num_bands - 1 else roi_y1
        band = main_mask[y0:y1, :]
        if band.size == 0:
            stats.append({
                'cover_ratio': 0.0,
                'left_cover': 0.0,
                'center_cover': 0.0,
                'right_cover': 0.0,
                'mean_width': 0.0,
                'centroid_x': 0.0,
                'y0': y0,
                'y1': y1,
            })
            continue

        cover_ratio = float(np.mean(band > 0))

        # 三分区覆盖率
        w3 = w // 3
        left_cover = float(np.mean(band[:, :w3] > 0))
        center_cover = float(np.mean(band[:, w3:2*w3] > 0))
        right_cover = float(np.mean(band[:, 2*w3:] > 0))

        # 平均行宽（像素宽度 / w）
        mean_width_vals = []
        centroid_vals = []
        for yy in range(y0, y1):
            row = main_mask[yy, :]
            xs = np.flatnonzero(row > 0)
            if xs.size > 0:
                width = (xs[-1] - xs[0] + 1) / float(w)
                mean_width_vals.append(width)
                centroid_vals.append((np.mean(xs) / float(w)) - 0.5)
        mean_width = float(np.mean(mean_width_vals)) if mean_width_vals else 0.0
        centroid_x = float(np.mean(centroid_vals)) if centroid_vals else 0.0

        stats.append({
            'cover_ratio': cover_ratio,
            'left_cover': left_cover,
            'center_cover': center_cover,
            'right_cover': right_cover,
            'mean_width': mean_width,
            'centroid_x': centroid_x,
            'y0': y0,
            'y1': y1,
        })

    return stats


def _try_skeleton_branching(main_mask, roi_y0, roi_y1):
    """
    可选：骨架提取，估计分叉点数量。若依赖不可用，返回 (False, 0, 0)。
    返回： (has_branch, branch_count, endpoint_count)
    """
    try:
        thinning = cv2.ximgproc.thinning  # 需要 opencv-contrib-python
    except Exception:
        return False, 0, 0

    roi = np.zeros_like(main_mask)
    roi[roi_y0:roi_y1, :] = main_mask[roi_y0:roi_y1, :]
    if np.count_nonzero(roi) == 0:
        return False, 0, 0

    skel = thinning((roi > 0).astype(np.uint8) * 255)

    # 统计 8 邻域度数
    skel_bin = (skel > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    # 用卷积近似计算邻居数：中心权重大，后续按阈值判断
    conv = cv2.filter2D(skel_bin, -1, kernel)

    # 粗略统计：邻居==1 视为端点；邻居>=3 视为分叉
    # 将中心像素值(10)剔除，邻居贡献为1
    # 对每个骨架像素，邻域和减去中心10后就是邻居个数
    ys, xs = np.where(skel_bin > 0)
    branch = 0
    endpoint = 0
    h, w = skel_bin.shape
    for y, x in zip(ys, xs):
        neigh_sum = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                neigh_sum += skel_bin[yy, xx]
        neigh_sum -= skel_bin[y, x]  # 去掉自身
        if neigh_sum == 1:
            endpoint += 1
        elif neigh_sum >= 3:
            branch += 1

    return (branch > 0), int(branch), int(endpoint)


def _infer_events_from_geometry(band_stats, gl, gc, gr):
    """
    根据分层几何特征推断事件。返回集合 set([(etype, edir), ...])。
    """
    events = set()
    if not band_stats:
        return events

    # 使用底部/中部/顶部 3 个参考层
    b0 = band_stats[0]
    bm = band_stats[len(band_stats)//2]
    bt = band_stats[-1]

    # 覆盖率/左右覆盖变化
    dl_left = bt['left_cover'] - b0['left_cover']
    dl_right = bt['right_cover'] - b0['right_cover']
    dl_center = bt['center_cover'] - b0['center_cover']

    # 宽度变化与质心漂移
    d_width = bt['mean_width'] - b0['mean_width']
    d_cx = bt['centroid_x'] - b0['centroid_x']

    # 基本阈值（可后续调参）
    EXPAND_THR = 0.15   # 一侧“上部”相对“底部”的覆盖增加幅度
    SYM_THR = 0.08      # 左右对称判定阈值
    CX_THR = 0.08       # 质心偏移阈值
    NARROW_THR = -0.12  # 宽度收窄阈值（负值）

    # 右转：右侧在上部显著扩张 + 质心右偏
    if (dl_right > EXPAND_THR and dl_right - dl_left > SYM_THR and d_cx > CX_THR and gc > 0.1):
        events.add(("TURN", "RIGHT"))

    # 左转：左侧在上部显著扩张 + 质心左偏
    if (dl_left > EXPAND_THR and dl_left - dl_right > SYM_THR and d_cx < -CX_THR and gc > 0.1):
        events.add(("TURN", "LEFT"))

    # 十字/大路口：两侧同时扩张明显
    if (dl_left > EXPAND_THR and dl_right > EXPAND_THR):
        events.add(("CROSS", "AHEAD"))

    # 丁字：前方中心明显变窄，而两侧仍然较宽（以中部为参考更稳）
    if (bm['center_cover'] < 0.08 and bm['left_cover'] > 0.18 and bm['right_cover'] > 0.18):
        events.add(("T_JUNCTION", "STOP_AHEAD"))

    # 中央障碍但可绕行：中心覆盖下降而至少一侧保持较高
    if (bt['center_cover'] < 0.08 and (bt['left_cover'] > 0.12 or bt['right_cover'] > 0.12)):
        events.add(("OBSTACLE_CENTER_BYPASSABLE", None))

    return events


def _draw_geom_debug(vis_img, contour, band_stats):
    if vis_img is None or not DEBUG_GEOM_VIS:
        return vis_img
    vis = vis_img.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
    # 画分层线与质心
    for b in band_stats:
        y0, y1 = b['y0'], b['y1']
        cv2.line(vis, (0, y0), (vis.shape[1], y0), (0, 255, 255), 1)
        cx = int((b['centroid_x'] + 0.5) * vis.shape[1])
        cy = (y0 + y1) // 2
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
    return vis


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

    在原有像素比例规则的基础上，加入形态学清理、最大连通域与分层几何分析，
    以更稳健地识别转弯/路口/障碍等事件。
    """
    events = []

    h, w = ground_mask.shape

    # ROI（下半部分代表近景区域）
    roi = ground_mask[int(h * 0.5):, :]  # 可根据实际情况调整 0.5 的比例

    # 左中右三块比例（保留原规则）
    left = roi[:, :w // 3]
    center = roi[:, w // 3: 2 * w // 3]
    right = roi[:, 2 * w // 3:]

    def ratio(region):
        return float(np.mean(region))

    gl = ratio(left)
    gc = ratio(center)
    gr = ratio(right)

    straight = gc > th_ground

    # 原始规则候选
    rule_events = []
    if straight and (gr - gl) > delta_side and gr > th_ground:
        rule_events.append(("TURN", "RIGHT"))
    if straight and (gl - gr) > delta_side and gl > th_ground:
        rule_events.append(("TURN", "LEFT"))
    if gc > th_ground and gl > th_ground and gr > th_ground:
        rule_events.append(("CROSS", "AHEAD"))
    if gc < th_ground and gl > th_ground and gr > th_ground:
        rule_events.append(("T_JUNCTION", "STOP_AHEAD"))
    if gc < th_ground and (gl > th_ground or gr > th_ground):
        rule_events.append(("OBSTACLE_CENTER_BYPASSABLE", None))

    # 形态学清理 + 主连通域 + 分层几何分析
    main_mask, contour = _clean_mask_and_extract_main_region(ground_mask)
    band_stats = []
    if contour is not None:
        roi_y0 = int(h * 0.5)
        roi_y1 = h
        band_stats = _compute_band_stats(main_mask, roi_y0, roi_y1, NUM_BANDS)
        geom_events = _infer_events_from_geometry(band_stats, gl, gc, gr)
    else:
        geom_events = set()

    # 合并事件并去重
    all_events = list({*rule_events, *geom_events})

    return all_events


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
# 7. 针对单张图片处理：推理 + 决策点 + 保存
############################

def process_single_image(model, img_path, save_dir):
    """
    对单张图片进行：
    1) 读取 & 分割
    2) 决策点识别
    3) 若有决策点，则保存到 save_dir，文件名加上决策点类型
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读取失败: {img_path}")
        return

    output=model.predict(input=img_path, target_size = -1)
    for res in output:
        pred =res.json
    pred =np.array(pred['res']['pred'][0])
    if not isinstance(pred, np.ndarray) or pred.ndim != 2:
        print(f"[ERROR] 预测结果 pred 不是 HxW 的 np.ndarray，请检查 model.predict 的返回格式.")
        return

    # ---------- 2. 获取地面 mask ----------
    ground_mask = get_ground_mask(pred)

    # ---------- 3. 检测决策点 ----------
    events = detect_outdoor_events(ground_mask)

    if not events:
        # 没有任何决策点，就不保存（根据你需求可改为仍然保存）
        print(f"[INFO] {os.path.basename(img_path)} 未识别到决策点，跳过保存")
        return

    # ---------- 4. 叠加可视化（可选） ----------
    vis = overlay_ground_mask(img, ground_mask)

    # 如果需要，叠加几何调试信息（轮廓、分层与质心）
    if DEBUG_GEOM_VIS:
        # 复用几何流程，避免重复计算
        main_mask, contour = _clean_mask_and_extract_main_region(ground_mask)
        if contour is not None:
            h, w = ground_mask.shape
            band_stats = _compute_band_stats(main_mask, int(h*0.5), h, NUM_BANDS)
            vis = _draw_geom_debug(vis, contour, band_stats)

    # 也可以在图像上写上事件文字方便调试
    y0 = 30
    for event in events:
        tag = event_to_tag(event)
        cv2.putText(vis, tag, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)
        y0 += 40

    # ---------- 5. 拼接输出文件名 ----------
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)

    # 可能出现多个事件类型，将它们的 tag 用 '_' 连接起来
    tags = [event_to_tag(e) for e in events]
    tags_str = "_".join(sorted(set(tags)))  # 去重并排序一下，避免重复

    out_name = f"{name}_{tags_str}{ext}"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)

    # ---------- 6. 保存结果图 ----------
    cv2.imwrite(out_path, vis)
    print(f"[SAVE] {out_path}")



def main():
    model_dir = "inference_model"  # TODO: 改成你的推理模型路径
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
