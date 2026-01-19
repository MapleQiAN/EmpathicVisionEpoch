# Sprint 1: 骨架构建 - 使用指南

## 📋 功能概述

Sprint 1 实现了从消防图到拓扑图的完整流程：

1. **透视变换**：自动检测并校正倾斜的消防图
2. **OCR文本提取**：识别地图上的房间号、楼梯、标识等文本
3. **骨架化**：将走廊线条提取为单像素骨架
4. **拓扑图构建**：生成节点（交叉点）和边（连接关系）
5. **OCR-节点关联**：将OCR文本自动关联到最近的图节点

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：PaddleOCR首次运行会自动下载模型，可能需要几分钟时间。

### 2. 启动后端服务

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后，访问 http://localhost:8000/docs 查看API文档。

### 3. 测试地图处理API

#### 方式1: 使用curl

```bash
curl -X POST "http://localhost:8000/map/process?apply_perspective=true&extract_ocr=true&auto_create_nodes=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/fire_map.jpg" \
  -F "building_id=building_001" \
  -F "floor_id=floor_4"
```

#### 方式2: 使用Python requests

```python
import requests

url = "http://localhost:8000/map/process"
files = {"file": open("fire_map.jpg", "rb")}
params = {
    "apply_perspective": True,
    "extract_ocr": True,
    "auto_create_nodes": True,
    "building_id": "building_001",
    "floor_id": "floor_4"
}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"节点数: {result['nodes_count']}")
print(f"边数: {result['edges_count']}")
print(f"OCR文本数: {result['ocr_count']}")
```

#### 方式3: 使用Swagger UI

1. 访问 http://localhost:8000/docs
2. 找到 `/map/process` 端点
3. 点击 "Try it out"
4. 上传图片文件，设置参数
5. 点击 "Execute"

## 📊 API响应格式

```json
{
  "success": true,
  "nodes_count": 25,
  "edges_count": 30,
  "ocr_count": 8,
  "data": {
    "nodes": [
      {"id": 0, "x": 100, "y": 200},
      {"id": 1, "x": 300, "y": 200},
      ...
    ],
    "edges": [
      {
        "source_node_id": 0,
        "target_node_id": 1,
        "length": 200.5,
        "weight": 200.5
      },
      ...
    ],
    "ocr_results": [
      {
        "text": "402",
        "center_x": 150,
        "center_y": 180,
        "bbox": [[140, 170], [160, 170], [160, 190], [140, 190]],
        "confidence": 0.95
      },
      ...
    ],
    "perspective_matrix": [[...], [...], [...]]
  }
}
```

## 🔧 参数说明

### `/map/process` 端点参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | File | 必填 | 消防图图片文件（jpg/png等） |
| `apply_perspective` | bool | true | 是否应用透视变换校正图像 |
| `extract_ocr` | bool | true | 是否提取OCR文本 |
| `building_id` | str | "default" | 建筑物ID |
| `floor_id` | str | "default" | 楼层ID |
| `auto_create_nodes` | bool | false | 是否自动创建图节点（将OCR关联到节点） |

## 🎯 使用场景示例

### 场景1: 仅提取拓扑结构（不处理OCR）

```python
params = {
    "apply_perspective": True,
    "extract_ocr": False,
    "auto_create_nodes": False
}
```

适用于：只需要走廊拓扑结构，不需要文本信息。

### 场景2: 完整处理（推荐）

```python
params = {
    "apply_perspective": True,
    "extract_ocr": True,
    "auto_create_nodes": True,
    "building_id": "building_001",
    "floor_id": "floor_4"
}
```

适用于：需要完整的拓扑图+文本标注+节点关联。

### 场景3: 手动处理OCR结果

```python
params = {
    "apply_perspective": True,
    "extract_ocr": True,
    "auto_create_nodes": False  # 先不自动创建，手动处理
}

# 获取结果后，手动处理OCR结果
result = response.json()["data"]
ocr_results = result["ocr_results"]
nodes = result["nodes"]

# 手动关联逻辑...
```

## 🐛 常见问题

### 1. PaddleOCR初始化失败

**问题**：`Failed to initialize PaddleOCR`

**解决**：
- 确保已安装：`pip install paddleocr`
- 首次运行会自动下载模型，需要网络连接
- 如果网络问题，可以手动下载模型到 `~/.paddleocr/`

### 2. 透视变换检测不到四个角

**问题**：`无法检测到四个角点`

**可能原因**：
- 图片质量差、对比度低
- 地图边界不清晰
- 图片已经被裁剪过

**解决**：
- 设置 `apply_perspective=False` 跳过透视变换
- 或手动标注四个角点（未来功能）

### 3. OCR识别率低

**问题**：OCR识别不到文本或识别错误

**解决**：
- 确保图片清晰度足够
- 透视变换后文本可能更清晰
- 可以调整PaddleOCR参数（在`MapDigitizer.__init__`中）

### 4. 节点数量过多/过少

**问题**：骨架化后节点数量不合理

**解决**：
- 调整`adaptive_block_size`、`adaptive_c`参数
- 调整`morph_kernel_size`参数
- 在`MapDigitizer`初始化时传入自定义参数

## 📝 下一步：Sprint 2

完成Sprint 1后，你将拥有：
- ✅ 拓扑图数据结构
- ✅ OCR文本标注
- ✅ 节点-文本关联

接下来可以开始Sprint 2：视觉指纹建立，将实际拍摄的照片与地图节点关联起来。

## 🔗 相关API

- `GET /graph` - 获取完整图结构
- `GET /graph/nodes` - 列出所有节点
- `GET /graph/edges` - 列出所有边
- `GET /graph/anchors` - 列出所有锚点（OCR关联）
- `POST /graph/nodes` - 手动创建节点
- `POST /graph/edges` - 手动创建边
