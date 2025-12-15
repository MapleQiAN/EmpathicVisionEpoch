import os
import cv2
import numpy as np
import argparse
import json
from paddlex import create_pipeline

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

# 判定稳定事件所需的帧数（这里只处理单张图片，可设为 1）
STABLE_FRAMES = 1

# 输入输出文件夹
DATA_DIR = "data"   # 输入图片目录
RES_DIR = "res"     # 输出结果目录


############################
# 2. 使用本地导出模型创建 PaddleX 语义分割管线（按官方文档方式）
############################

def _find_seg_config(path: str) -> str:
    """
    寻找导出目录中的管线配置文件（PaddleX 文档推荐用 config 方式创建管线）：
    - deploy.yaml / pipeline.yaml / inference.yml / infer_cfg.yml / inference.json
    若传入的 path 本身就是一个文件，则直接返回。
    否则在目录下按以上优先级搜索，找到则返回其完整路径，否则抛错。
    """
    if os.path.isfile(path):
        return path

    candidates = [
        os.path.join(path, "deploy.yaml"),
        os.path.join(path, "pipeline.yaml"),
        os.path.join(path, "inference.yml"),
        os.path.join(path, "infer_cfg.yml"),
        os.path.join(path, "inference.json"),
    ]
    for cfg in candidates:
        if os.path.exists(cfg):
            return cfg
    raise FileNotFoundError(
        f"未找到可用的管线配置文件，请检查导出目录是否完整: {path}\n"
        f"期望存在的文件之一: deploy.yaml / pipeline.yaml / inference.yml / infer_cfg.yml / inference.json"
    )


def _load_config_to_dict(cfg_path: str) -> dict:
    """
    将配置文件内容加载为 dict。
    - 支持 .yaml/.yml（需安装 pyyaml）、.json
    - 若为不支持后缀或解析失败则抛错
    """
    ext = os.path.splitext(cfg_path)[1].lower()
    if ext in [".yaml", ".yml"]:
        try:
            import yaml  # 延迟导入，避免环境无 pyyaml 时影响其它逻辑
        except ImportError as e:
            raise ImportError(
                "解析 YAML 失败：未安装 pyyaml，请先安装：pip install pyyaml"
            ) from e
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"YAML 配置解析后不是字典类型: {type(cfg)}")
        return cfg
    elif ext == ".json":
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"JSON 配置解析后不是字典类型: {type(cfg)}")
        return cfg
    else:
        # 部分版本也可能是其它命名，但通常内容仍为 yaml，尝试 yaml 解析作为兜底
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError
            return cfg
        except Exception as e:
            raise ValueError(
                f"无法解析配置文件（仅支持 YAML/JSON）: {cfg_path}\n"
                f"请改用 deploy.yaml/pipeline.yaml/inference.yml/infer_cfg.yml 或 inference.json"
            ) from e


def _make_paths_absolute(obj, base_dir: str):
    """
    递归地将配置 dict/list 中的相对路径转成绝对路径：
    - 跳过以 http(s):// 开头的 URL
    - 仅当 join(base_dir, value) 存在时才替换
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _make_paths_absolute(v, base_dir)
        return obj
    elif isinstance(obj, list):
        return [_make_paths_absolute(v, base_dir) for v in obj]
    elif isinstance(obj, str):
        v = obj
        if v.startswith("http://") or v.startswith("https://"):
            return v
        if os.path.isabs(v):
            return v
        cand = os.path.normpath(os.path.join(base_dir, v))
        if os.path.exists(cand):
            return cand
        return v
    else:
        return obj


def load_seg_pipeline_local(model_path: str):
    """
    按 PaddleX 文档推荐：通过本地导出的 config 文件来创建语义分割管线，
    确保不会使用 PaddleX 内置/在线模型。

    参考文档: https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/semantic_segmentation.html
    """
    cfg_path = _find_seg_config(model_path)

    # 读取配置为 dict，按你当前 PaddleX 版本的 create_pipeline 期望传入 dict
    config_dict = _load_config_to_dict(cfg_path)
    if not isinstance(config_dict, dict):
        raise TypeError("配置对象不是字典，无法用于 create_pipeline(config=...)")

    # 尝试将其中的相对路径改为绝对路径（相对 cfg 所在目录）
    base_dir = os.path.dirname(os.path.abspath(cfg_path))
    config_dict = _make_paths_absolute(config_dict, base_dir)

    # 如果导出配置里未包含 pipeline_name，补充默认值避免内部取键报错
    if "pipeline_name" not in config_dict:
        config_dict["pipeline_name"] = "semantic_segmentation"

    # 先尝试新版 API：仅传 config=dict
    try:
        pipeline = create_pipeline(config=config_dict)
        print(f"[INFO] 已使用本地配置创建分割管线: {cfg_path}")
        return pipeline
    except Exception as e1:
        # 兼容某些版本：需要显式传 pipeline 名称（同时传 dict）
        try:
            pipeline = create_pipeline(pipeline="semantic_segmentation", config=config_dict)
            print(f"[INFO] 已使用本地配置创建分割管线(兼容模式1): {cfg_path}")
            return pipeline
        except Exception as e2:
            # 极少数版本参数名不同（pipeline_config）
            try:
                pipeline = create_pipeline(pipeline="semantic_segmentation", pipeline_config=config_dict)
                print(f"[INFO] 已使用本地配置创建分割管线(兼容模式2): {cfg_path}")
                return pipeline
            except Exception as e3:
                raise RuntimeError(
                    "创建 PaddleX 语义分割管线失败，请检查 PaddleX 版本与导出配置是否匹配。\n"
                    f"cfg: {cfg_path}\n"
                    f"错误链: {repr(e1)} -> {repr(e2)} -> {repr(e3)}"
                )


def get_ground_mask(pred):
    """
    pred: H x W 的类别 id 图 (int)
    返回: H x W 的 bool 数组，True 表示地面 (road/sidewalk)
    """
    ground_mask = np.isin(pred, GROUND_IDS)
    return ground_mask


def simple_refine_ground_mask(ground_mask_bool,
                              kernel_rel=0.008,
                              min_area_ratio=0.0005,
                              keep_only_bottom_connected=False,
                              hole_max_area_ratio=0.0):
    """
    ground_mask_bool: HxW 的bool初始地面掩码
    kernel_rel: 形态学核大小相对最短边的比例(0.008~0.02常用)
    min_area_ratio: 连通域最小面积占比(0.0005~0.003常用)
    keep_only_bottom_connected: 仅保留与底边相连的连通域
    """
    H, W = ground_mask_bool.shape
    b = (ground_mask_bool.astype(np.uint8) * 255)

    # 形态学：先闭后开（填缝、去小噪）
    k = max(3, int(min(H, W) * float(kernel_rel)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=1)

    # 连通域过滤
    num, labels, stats, _ = cv2.connectedComponentsWithStats((b > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (b > 0)

    area_th = H * W * float(min_area_ratio)
    keep = np.zeros((H, W), dtype=np.uint8)

    # 与底边相连判定
    def touches_bottom(lbl_id):
        return np.any(labels[-1, :] == lbl_id) or np.any(labels[-2, :] == lbl_id)

    kept_ids = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= area_th:
            if not keep_only_bottom_connected or touches_bottom(i):
                keep[labels == i] = 255
                kept_ids.append(i)

    # 如果严格与底边相连后一个都没保留，兜底保留最大连通域
    if keep_only_bottom_connected and len(kept_ids) == 0:
        i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        keep[labels == i] = 255

    # 轻微平滑边界
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 可选：填充小“空洞”（如小树叶遮挡）以避免误判为障碍
    if hole_max_area_ratio and hole_max_area_ratio > 0:
        mask_bin = (keep > 0).astype(np.uint8)  # 1: ground, 0: background
        inv = (255 - mask_bin * 255).copy()     # 255: background, 0: ground
        flood_mask = np.zeros((H + 2, W + 2), np.uint8)  # floodFill 需要比图像大2的掩码
        cv2.floodFill(inv, flood_mask, (0, 0), 0)       # 去掉与外边界连通的背景
        holes = (inv == 255)                              # 剩下的255即为地面内部“洞”

        num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        hole_area_th = H * W * float(hole_max_area_ratio)
        for i_h in range(1, num_h):
            if stats_h[i_h, cv2.CC_STAT_AREA] <= hole_area_th:
                keep[labels_h == i_h] = 255

    return (keep > 0)

############################
# 4. 根据 ground_mask 计算决策点事件（单张图）
############################

def suppress_small_roi_obstacles(ground_mask,
                                 obs_min_area_ratio=0.01,
                                 obs_min_width_ratio=0.06,
                                 roi_top_rel=0.5):
    """
    在事件检测前对 ground_mask 做一次“误障碍抑制”：
    - 仅针对 ROI(下半部分) 内的 非地面 斑块做处理；
    - 将面积很小 或 水平宽度很窄 的斑块视为树叶等小遮挡，直接填回地面。

    参数:
    ground_mask: HxW bool, True 为地面
    obs_min_area_ratio: 小斑块面积阈值(相对 ROI 面积)
    obs_min_width_ratio: 小斑块最小宽度(相对整图宽度)
    roi_top_rel: ROI 顶部相对高度, 默认 0.5 即下半幅
    """
    H, W = ground_mask.shape
    top = int(H * float(roi_top_rel))
    roi = ground_mask[top:, :].copy()

    # 非地面区域
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
            # 将该小障碍斑块恢复为地面
            roi[labels == i] = True

    out = ground_mask.copy()
    out[top:, :] = roi
    return out


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
# 6. 解析 PaddleX predict 输出为 HxW 类别图
############################

def _to_hxw_pred_array(data):
    """将多种可能的数据结构转换为 HxW 的 np.ndarray(int)。失败则抛错。"""
    arr = None
    # 常见：list[list[...] ] 或 np.ndarray
    if isinstance(data, np.ndarray):
        arr = data
    else:
        try:
            arr = np.array(data)
        except Exception:
            arr = None
    if arr is None:
        raise ValueError("无法将预测结果转换为 ndarray")

    # 兼容可能的形状： (H,W), (1,H,W), (H,W,1)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"预测数组维度不为2，实际形状: {arr.shape}")

    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)
    return arr


def parse_pred_from_output(output):
    """
    尝试从 PaddleX pipeline 的 predict 返回结果中解析出 HxW 的类别图。
    兼容多种字段命名：'res'/'result' 下的 'pred'/'label_map'/'seg_map' 等。
    """
    # 取到一个 dict
    data = None
    # 1) output 可能是迭代器/列表，元素带 .json
    try:
        for res in output:
            if hasattr(res, 'json'):
                data = res.json
                break
    except TypeError:
        # output 不可迭代
        pass

    # 2) 如果没取到，可能本身就是 dict
    if data is None and isinstance(output, dict):
        data = output

    if data is None:
        raise ValueError("无法从 predict 的输出中获取字典数据，请打印 output 检查具体结构")

    # 常见结构 1： data['res']['pred'][0]
    try:
        if 'res' in data and isinstance(data['res'], dict) and 'pred' in data['res']:
            return _to_hxw_pred_array(data['res']['pred'][0])
    except Exception:
        pass

    # 常见结构 2： data['result']['label_map'] 或 ['pred'] 或 ['seg_map']
    if 'result' in data and isinstance(data['result'], dict):
        cand_keys = ['label_map', 'pred', 'seg_map']
        for k in cand_keys:
            if k in data['result']:
                try:
                    v = data['result'][k]
                    # 有些会是 [HxW] 或 [[HxW]]
                    if isinstance(v, list) and len(v) == 1:
                        v = v[0]
                    return _to_hxw_pred_array(v)
                except Exception:
                    continue

    # 常见结构 3： 顶层就有 'pred'/'label_map'/'seg_map'
    for k in ['pred', 'label_map', 'seg_map']:
        if k in data:
            v = data[k]
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            return _to_hxw_pred_array(v)

    # 若仍未解析成功，抛出并让用户打印 data
    raise KeyError(
        f"未能在 predict 返回中找到可用的类别图字段，可用键: {list(data.keys())}"
    )


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

    # 按 PaddleX 文档，predict 入参通常支持图像路径或 ndarray，这里仍传路径
    output = model.predict(input=img_path, target_size=-1)

    try:
        pred = parse_pred_from_output(output)
    except Exception as e:
        print("[ERROR] 解析预测结果失败: ", repr(e))
        # 尝试打印一次结构帮助定位
        try:
            preview = None
            try:
                for res in output:
                    if hasattr(res, 'json'):
                        preview = list(res.json.keys()) if isinstance(res.json, dict) else type(res.json)
                        break
            except TypeError:
                if isinstance(output, dict):
                    preview = list(output.keys())
            print("[DEBUG] predict 返回预览: ", preview)
        except Exception:
            pass
        return

    if not isinstance(pred, np.ndarray) or pred.ndim != 2:
        print(f"[ERROR] 预测结果 pred 不是 HxW 的 np.ndarray，请检查 model.predict 的返回格式.")
        return

    # ---------- 2. 获取地面 mask ----------
    ground_mask = simple_refine_ground_mask(
        get_ground_mask(pred),
        kernel_rel=0.008,
        min_area_ratio=0.0005,
        keep_only_bottom_connected=True,
        hole_max_area_ratio=HOLE_MAX_AREA_RATIO
    )

    # 针对事件检测：抑制 ROI 内很小且很窄的“非地面”斑块（如树叶）
    ground_mask_evt = suppress_small_roi_obstacles(
        ground_mask,
        obs_min_area_ratio=OBS_MIN_AREA_RATIO,
        obs_min_width_ratio=OBS_MIN_WIDTH_RATIO
    )

    # ---------- 3. 检测决策点 ----------
    events = detect_outdoor_events(ground_mask_evt)

    if not events:
        # 没有任何决策点，就不保存（根据你需求可改为仍然保存）
        print(f"[INFO] {os.path.basename(img_path)} 未识别到决策点，跳过保存")
        return

    # ---------- 4. 叠加可视化（可选） ----------
    vis = overlay_ground_mask(img, ground_mask_evt)

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
    parser = argparse.ArgumentParser(description="使用 PaddleX 本地导出模型进行语义分割并输出带事件标注的可视化结果")
    parser.add_argument(
        "--model",
        type=str,
        default="inference_model/OCRNet_HRNet-W48_infer",
        help="本地导出模型目录或配置文件(deploy.yaml/pipeline.yaml/inference.yml 等)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help="输入图片目录"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=RES_DIR,
        help="输出结果目录"
    )
    args = parser.parse_args()

    # 使用你下载/导出的本地模型（严格按照文档通过 config 创建管线）
    model = load_seg_pipeline_local(args.model)

    os.makedirs(args.out_dir, exist_ok=True)

    # 遍历目录下的所有图片文件
    for fname in os.listdir(args.data_dir):
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # 简单筛选一下常见图片后缀
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        print(f"[PROCESS] {fpath}")
        process_single_image(model, fpath, args.out_dir)


if __name__ == "__main__":
    main()
