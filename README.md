# 项目说明：语义分割与户外通行辅助示例

本项目包含两个脚本，基于 PaddleX 的语义分割能力：
- demo.py：使用 PaddleX 预置的语义分割 pipeline 对单张图片进行预测，保存可视化结果与 JSON。
- process_images.py：加载本地导出的 PaddleX 语义分割推理模型，遍历 data/ 下的图片，提取“可行走地面”并进行简单的通行事件检测（如转弯、十字、丁字、中央可绕行障碍），保存带可视化覆盖和事件标签的结果图到 res/。

## 目录

- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [事件检测规则概览](#事件检测规则概览)
- [算法流程图](#算法流程图)
- [可视化与保存行为](#可视化与保存行为)
- [常见问题排查](#常见问题排查)
- [许可与数据](#许可与数据)
- [致谢](#致谢)

## 目录结构

- demo.py：单图演示脚本（自动下载/使用 PaddleX 预置模型）。
- process_images.py：批处理 + 事件检测脚本（需要本地推理模型目录）。
- label.txt：示例标签映射（Cityscapes 19 类：0=road，1=sidewalk，…）。
- demo.jpg：演示用示例图片。
- data/：请自行创建，放置待处理图片（.jpg/.jpeg/.png/.bmp）。
- res/：脚本运行后输出结果目录（自动创建）。
- output/：demo.py 运行后保存图片和 JSON 的目录（自动创建）。

## 环境依赖

- Python 3.8+（建议）
- 主要依赖：
  - paddlex
  - opencv-python（或在无桌面环境使用 opencv-python-headless）
  - numpy

安装示例：

```
pip install paddlex opencv-python numpy
# 服务器/无 GUI 环境可用：
# pip install paddlex opencv-python-headless numpy
```

> 提示：PaddleX 的具体版本与 API 可能存在差异。process_images.py 中已做了多版本兼容处理（model、model_dir、config、pipeline_config 等）。

## 快速开始

### 1) 单图演示（demo.py）

无需自备模型，直接运行：

```
python demo.py
```

脚本会：
- 使用 `create_pipeline(pipeline="semantic_segmentation")` 创建分割 pipeline；
- 对 demo.jpg 进行预测；
- 将可视化结果与 JSON 保存到 `./output/` 目录下。

### 2) 批处理 + 事件检测（process_images.py）

1. 准备推理模型目录（PaddleX 导出）：
   - 在脚本中默认路径为：
     ```
     model_dir = "inference_model/OCRNet_HRNet-W48_infer"
     ```
   - 请将其修改为你的实际推理模型目录。该目录通常包含（不同版本命名略有差异）：
     - 模型文件：`model.pdmodel`、`model.pdiparams`（或同等物）
     - 配置文件：`inference.yml`、`deploy.yaml`、`pipeline.yaml`、`infer_cfg.yml` 或 `inference.json` 之一

2. 准备输入图片：
   - 在项目根目录下创建 `data/` 文件夹，将待处理的 `.jpg/.jpeg/.png/.bmp` 图片放入其中。

3. 运行脚本：
   ```
   python process_images.py
   ```

4. 查看结果：
   - 输出会保存在 `res/` 下，文件名会在原图名基础上追加识别到的事件标签，例如：
     - `street_001_TURN_LEFT.jpg`
     - `park_017_T_JUNCTION_OBSTACLE_CENTER_BYPASSABLE.jpg`
   - 图像中会以半透明绿色覆盖“可行走地面”，并在左上角写上事件类型。

## 事件检测规则概览

脚本将图像下半部分作为 ROI（视为行走区域），并将其横向划分为左/中/右三块，统计每块区域内“地面”像素比例（基于分割结果中 0:road、1:sidewalk）。核心阈值与规则在 process_images.py 顶部可调：

- GROUND_IDS = [0, 1]：视为可行走地面的类别 id（Cityscapes 的 road/sidewalk）。
- TH_GROUND（默认 0.15）：某区域地面比例超过该值，认为该区域“有明显地面”。
- DELTA_SIDE（默认 0.10）：左右地面比例差异达到该值，认为一侧出现“明显通路”。
- HOLE_MAX_AREA_RATIO（默认 0.0015）：填充地面中的细小“空洞”（如树叶遮挡）的面积上限占整图比例，用于减少误判为障碍。
- OBS_MIN_AREA_RATIO（默认 0.01）：抑制 ROI 内非常小的“非地面”斑块（视为小遮挡）。
- OBS_MIN_WIDTH_RATIO（默认 0.06）：同上，对应最小水平宽度阈值。

基于上述统计与抑制，脚本输出以下事件（可能同时出现多个）：
- TURN_LEFT / TURN_RIGHT：在保持前进可行的前提下，一侧地面显著增多，提示可转向。
- CROSS：左/中/右三块均有较多地面，可能为十字路口或大开口区域。
- T_JUNCTION（显示为 STOP_AHEAD 提示）：中间地面少而两侧多，可能为丁字路口，建议前方注意。
- OBSTACLE_CENTER_BYPASSABLE：中央地面不足，但至少一侧可绕行。

可根据场景对阈值进行微调以取得更稳定的识别效果。

## 算法流程图

下面给出 process_images.py 的核心处理流程（含分割、后处理与事件判定）。可在支持 Mermaid 的渲染器（如 GitHub/GitLab/VS Code 插件）中直接预览。

```mermaid
flowchart TD
    A[开始] --> B[加载配置与阈值]
    B --> C{模式选择}
    C -->|demo.py| D[create_pipeline 语义分割]
    C -->|process_images.py| E[加载本地推理模型]
    D --> F[读取输入图像]
    E --> F
    F --> G[语义分割推理 -> 类别图 seg_map]
    G --> H[生成地面掩码 mask\n(类别ID ∈ {0,1})]
    H --> I[形态学与抑制\n- 填补细小空洞\n- 去除过小/过窄障碍]
    I --> J[截取 ROI: 图像下半部分]
    J --> K[横向划分 ROI: 左/中/右]
    K --> L[统计每块地面比例 pL,pC,pR]
    L --> M{事件判定}
    M -->|pC ≥ TH 且 pL - pC ≥ Δ| N[TURN_LEFT]
    M -->|pC ≥ TH 且 pR - pC ≥ Δ| O[TURN_RIGHT]
    M -->|pL ≥ TH 且 pC ≥ TH 且 pR ≥ TH| P[CROSS]
    M -->|pC < TH 且 (pL ≥ TH 或 pR ≥ TH)| Q[OBSTACLE_CENTER_BYPASSABLE]
    M -->|pC < TH 且 pL ≥ TH 且 pR ≥ TH| R[T_JUNCTION / STOP_AHEAD]
    N --> S[叠加可视化与事件标签]
    O --> S
    P --> S
    Q --> S
    R --> S
    S --> T[保存至 res/ 并依据事件重命名]
    T --> U[结束]
```

流程要点：
- 分割输出解析为 H×W 类别图，再转为“地面”二值掩码。
- 通过小洞填充与小障碍抑制，降低噪声影响。
- ROI 仅取下半部分，更贴近日常行走视野；横向三分便于侧向通路判断。
- 根据阈值 TH_GROUND 与差异 DELTA_SIDE 等进行事件判定，可多事件并存。

## 可视化与保存行为

- demo.py：
  - 使用 res.print() 输出信息；
  - 使用 res.save_to_img(save_path="./output/") 保存图像结果；
  - 使用 res.save_to_json(save_path="./output/") 保存 JSON 结果。

- process_images.py：
  - 对每张图片：
    1) 运行分割，解析为 H×W 的类别图；
    2) 生成地面掩码并做简单形态学/小障碍抑制；
    3) 检测事件；
    4) 将地面覆盖为半透明绿色，叠加事件标签文本；
    5) 依据事件标签重命名并保存到 `res/`。

## 常见问题排查

1. 无法加载模型（load_seg_model 报错）：
   - 请确认传入的 `model_dir` 路径正确且包含模型与推理配置文件。
   - 脚本会尝试多种 PaddleX 版本常见的参数与配置文件名（`model`、`model_dir`、`config`、`pipeline_config`；`inference.yml`、`deploy.yaml`、`pipeline.yaml`、`infer_cfg.yml`、`inference.json`）。
   - 若仍报错，可打印/检查你当前的 PaddleX 版本与导出产物结构。

2. 预测结果解析失败（parse_pred_from_output 报错）：
   - 脚本已兼容常见字段（`res.pred`、`result.label_map/pred/seg_map` 等）。
   - 若你的返回结构不同，可在错误信息后打印的 preview 提示基础上自行适配。

3. 读图失败：
   - OpenCV `cv2.imread` 返回 None，通常是路径不对或文件损坏；请检查 `data/` 下的文件是否存在且可读。

4. 事件识别效果不稳：
   - 可适当上调/下调 `TH_GROUND`、`DELTA_SIDE`，并调整抑制参数（`HOLE_MAX_AREA_RATIO`、`OBS_MIN_AREA_RATIO`、`OBS_MIN_WIDTH_RATIO`）。
   - 也可修改 ROI 高度比例（如将 `roi = ground_mask[int(h*0.5):, :]` 中的 0.5 调整为 0.55/0.6 等）。

## 许可与数据

- 标签映射参考 Cityscapes 19 类（见 label.txt）。
- 本仓库未包含第三方数据与模型，请按各自许可获取并使用。

## 致谢

- [PaddleX](https://github.com/PaddlePaddle/PaddleX)
- Cityscapes 数据集标签定义

---

如需进一步定制（如增加语音提示/路径规划/录像流输入），欢迎提出需求。
