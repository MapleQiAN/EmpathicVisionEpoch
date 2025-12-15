import os
import cv2
import numpy as np
import argparse
import json
from paddlex import create_pipeline, create_model

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

# 输入输出文件夹默认值
DATA_DIR = "data"   # 输入图片目录
RES_DIR = "res"     # 输出结果目录


############################
# 2A. 使用本地导出配置创建 PaddleX 语义分割管线
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
        # 兜底尝试按 YAML 解析
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


def load_seg_pipeline_local(config_path: str, device: str | None = None):
    """
    通过本地导出的 config 文件来创建语义分割管线，
    并可选强制指定推理设备（覆盖配置文件中的 device）。
    """
    cfg_path = _find_seg_config(config_path)

    # 读取配置为 dict
    config_dict = _load_config_to_dict(cfg_path)

    # 路径转为绝对路径
    base_dir = os.path.dirname(os.path.abspath(cfg_path))
    config_dict = _make_paths_absolute(config_dict, base_dir)

    # 补充/覆盖设备
    if device:
        config_dict["device"] = device

    # 补充管线名称，避免内部取键报错
    if "pipeline_name" not in config_dict:
        config_dict["pipeline_name"] = "semantic_segmentation"

    # 先尝试仅传 config=dict
    last_err = None
    try:
        pipeline = create_pipeline(config=config_dict)
        print(f"[INFO] 已使用本地配置创建分割管线: {cfg_path}")
        return pipeline
    except Exception as e:
        last_err = e
    # 兼容传 pipeline 名称
    try:
        pipeline = create_pipeline(pipeline="semantic_segmentation", config=config_dict)
        print(f"[INFO] 已使用本地配置创建分割管线(兼容模式1): {cfg_path}")
        return pipeline
    except Exception as e:
        last_err = e
    # 兼容 pipeline_config 命名
    try:
        pipeline = create_pipeline(pipeline="semantic_segmentation", pipeline_config=config_dict)
        print(f"[INFO] 已使用本地配置创建分割管线(兼容模式2): {cfg_path}")
        return pipeline
    except Exception as e:
        last_err = e
        raise RuntimeError(
            "创建 PaddleX 语义分割管线失败，请检查 PaddleX 版本与导出配置是否匹配。\n"
            f"cfg: {cfg_path}\n"
            f"错误: {repr(last_err)}"
        )


############################
# 2B. 使用 create_model 方式加载特定模型与设备
############################

def _parse_target_size(arg_val: str | None):
    """
    将命令行传入的 target_size 字符串解析为 int/tuple/None：
    - None 或 "none" -> None
    - "-1" -> -1
    - "512" -> 512
    - "512,512" 或 "512x512" 或 "512*512" -> (512, 512)
    """
    if arg_val is None:
        return None
    s = str(arg_val).strip().lower()
    if s in ["", "none", "null"]:
        return None
    if s == "-1":
        return -1
    # tuple 形式
    for sep in [",", "x", "*"]:
        if sep in s:
            a, b = s.split(sep, 1)
            return (int(a.strip()), int(b.strip()))
    # 单 int
    return int(s)


def load_seg_model_api(model_name: str, model_dir: str | None, device: str, target_size, use_hpip: bool, hpi_config: dict | None):
    """
    按文档使用 create_model 直接创建模型：可指定 model_name、model_dir、device、target_size 等。
    - 指定 model_name 后，若同时提供 model_dir，会从该目录加载你的自定义权重。
    - device 支持 "gpu:0"/"cpu"/"npu:0" 等。
    """
    kwargs = {
        "model_name": model_name,
        "device": device,
        "use_hpip": use_hpip,
    }
    if model_dir:
        kwargs["model_dir"] = model_dir
    if target_size is not None:
        kwargs["target_size"] = target_size
    if hpi_config is not None:
        kwargs["hpi_config"] = hpi_config

    model = create_model(**kwargs)
    print(f"[INFO] 已通过 create_model 加载模型: name={model_name}, dir={model_dir}, device={device}, target_size={target_size}")
    return model


############################
# 3. 业务逻辑（ground_mask 处理/事件判定/可视化）
############################

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
    H, W = ground_mask_bool.shape
    b = (ground_mask_bool.astype(np.uint8) * 255)

    # 形态学：先闭后开
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

    vis[mask_3c] = cv2.addWeighted(
        vis, 1 - alpha, green, alpha, 0
    )[mask_3c]
    return vis


############################
# 4. 解析 PaddleX predict 输出为 HxW 类别图
############################

def _to_hxw_pred_array(data):
    arr = None
    if isinstance(data, np.ndarray):
        arr = data
    else:
        try:
            arr = np.array(data)
        except Exception:
            arr = None
    if arr is None:
        raise ValueError("无法将预测结果转换为 ndarray")

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
    兼容多版本的返回结构，解析出 HxW 的类别图。
    优先尝试 data['res']['pred'][0]，也尝试 data['result'] 下的 label_map/pred/seg_map。
    """
    data = None
    try:
        for res in output:
            if hasattr(res, 'json'):
                data = res.json
                break
    except TypeError:
        pass

    if data is None and isinstance(output, dict):
        data = output

    if data is None:
        raise ValueError("无法从 predict 的输出中获取字典数据，请打印 output 检查具体结构")

    try:
        if 'res' in data and isinstance(data['res'], dict) and 'pred' in data['res']:
            return _to_hxw_pred_array(data['res']['pred'][0])
    except Exception:
        pass

    if 'result' in data and isinstance(data['result'], dict):
        for k in ['label_map', 'pred', 'seg_map']:
            if k in data['result']:
                v = data['result'][k]
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                return _to_hxw_pred_array(v)

    for k in ['pred', 'label_map', 'seg_map']:
        if k in data:
            v = data[k]
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            return _to_hxw_pred_array(v)

    raise KeyError(
        f"未能在 predict 返回中找到可用的类别图字段，可用键: {list(data.keys())}"
    )


############################
# 5. 单张图推理 + 决策点识别 + 保存
############################

def process_single_image(model, img_path, save_dir, predict_target_size=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读取失败: {img_path}")
        return

    # 如果未指定 predict_target_size，则不传该参数，沿用模型内设置
    if predict_target_size is None:
        output = model.predict(input=img_path)
    else:
        output = model.predict(input=img_path, target_size=predict_target_size)

    try:
        pred = parse_pred_from_output(output)
    except Exception as e:
        print("[ERROR] 解析预测结果失败: ", repr(e))
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
    parser = argparse.ArgumentParser(description="使用 PaddleX 模型/管线进行语义分割并输出带事件标注的可视化结果")

    # 运行模式
    parser.add_argument("--mode", choices=["pipeline", "model"], default="pipeline", help="加载方式：pipeline(本地配置) 或 model(create_model)")

    # pipeline 模式参数
    parser.add_argument("--config", type=str, default="inference_model/OCRNet_HRNet-W48_infer", help="本地导出模型目录或配置文件(deploy.yaml/pipeline.yaml/inference.yml 等)")

    # model 模式参数
    parser.add_argument("--model_name", type=str, default=None, help="模型名称（如 PP-LiteSeg-T 等）。model 模式必填")
    parser.add_argument("--model_dir", type=str, default=None, help="自定义权重/模型的本地目录，可选。若不填则使用内置权重")

    # 通用推理设备与尺寸
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 gpu:0 / cpu / npu:0。pipeline 模式未填则使用配置文件内设置")
    parser.add_argument("--target_size", type=str, default=None, help="推理分辨率。示例：-1 或 512 或 512,512")

    # 高性能推理插件（仅 model 模式有效）
    parser.add_argument("--use_hpip", action="store_true", help="是否启用高性能推理插件(仅 model 模式)")
    parser.add_argument("--hpi_config", type=str, default=None, help="高性能推理配置(JSON 字符串或 JSON 文件路径，仅 model 模式)")

    # 数据与输出
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="输入图片目录")
    parser.add_argument("--out_dir", type=str, default=RES_DIR, help="输出结果目录")

    args = parser.parse_args()

    # 解析 target_size
    ts = _parse_target_size(args.target_size)

    # 解析 hpi_config
    hpi_cfg = None
    if args.hpi_config:
        if os.path.isfile(args.hpi_config):
            with open(args.hpi_config, "r", encoding="utf-8") as f:
                hpi_cfg = json.load(f)
        else:
            try:
                hpi_cfg = json.loads(args.hpi_config)
            except Exception:
                print("[WARN] hpi_config 既不是文件也不是合法 JSON 字符串，已忽略。")
                hpi_cfg = None

    # 构建模型/管线
    if args.mode == "model":
        if not args.model_name:
            raise ValueError("model 模式下必须指定 --model_name")
        model = load_seg_model_api(
            model_name=args.model_name,
            model_dir=args.model_dir,
            device=(args.device or "gpu:0"),
            target_size=ts,
            use_hpip=args.use_hpip,
            hpi_config=hpi_cfg,
        )
    else:
        # pipeline 模式：从本地配置创建，允许 --device 覆盖
        model = load_seg_pipeline_local(args.config, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    # 遍历目录下的所有图片文件
    for fname in os.listdir(args.data_dir):
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        print(f"[PROCESS] {fpath}")
        process_single_image(model, fpath, args.out_dir, predict_target_size=ts)


if __name__ == "__main__":
    main()
