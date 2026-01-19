# MVP 实现总结

## 实现概述

已完成楼层消防图处理的MVP版本，能够扫描楼层消防图并转化为可处理的数据结构，识别并分类重要的节点（路口、转角、重要公共设施等）。

## 核心功能

### 1. 节点分类系统

实现了四种节点类型的自动识别：

- **路口 (Intersection)**: 连接3条或以上路径的交叉点
  - 识别方法：节点度数 ≥ 3
  - 置信度：0.85

- **转角 (Corner)**: 连接2条路径且角度接近90度的节点
  - 识别方法：节点度数 = 2 且角度在 90° ± 容差范围内
  - 默认容差：30度（可配置）
  - 置信度：0.8

- **端点 (Endpoint)**: 只连接1条路径的节点
  - 识别方法：节点度数 = 1
  - 置信度：0.9

- **重要设施 (Facility)**: 通过OCR识别的重要公共设施
  - 识别方法：OCR文本关键词匹配 + 距离关联
  - 置信度：0.95（有OCR支持时）

### 2. 重要设施识别

支持识别以下类型的设施：

- **楼梯 (stairs)**: 楼梯、楼梯间、安全楼梯、疏散楼梯
- **电梯 (elevator)**: 电梯、升降机
- **安全出口 (exit)**: 安全出口、出口、疏散出口
- **消防栓 (fire_hydrant)**: 消防栓、消火栓、消防、灭火器
- **卫生间 (toilet)**: 厕所、洗手间、卫生间、盥洗室
- **房间 (room)**: 房间号（数字或数字+字母组合）

### 3. OCR-节点关联

- 自动将OCR识别结果与最近的图节点关联
- 距离阈值：100像素（可配置）
- 支持中英文OCR识别
- 保留OCR文本、置信度、边界框等信息

### 4. 数据结构扩展

节点数据结构现在包含：

```python
{
    "id": 0,                    # 节点ID
    "x": 100,                   # X坐标
    "y": 200,                   # Y坐标
    "node_type": "intersection", # 节点类型
    "degree": 3,                # 节点度数
    "confidence": 0.85,          # 识别置信度
    "angle": 87.5,              # 转角角度（仅转角节点）
    "facility_type": "stairs",   # 设施类型（仅设施节点）
    "ocr_text": "楼梯间",        # OCR文本（如果有关联）
    "ocr_confidence": 0.95      # OCR置信度（如果有关联）
}
```

## 技术栈选择

### 当前实现（Python后端）

- **图像处理**: OpenCV (cv2)
- **图结构处理**: NetworkX
- **OCR识别**: PaddleOCR
- **图像骨架化**: scikit-image
- **Web框架**: FastAPI

### Android迁移方案

#### 方案1: REST API调用（推荐用于MVP）

**架构**:
```
Android App → HTTP API → Python后端 → 处理结果
```

**优点**:
- 实现简单，快速上线
- 无需在Android端部署模型
- 易于维护和更新算法
- 可以集中处理，节省移动端资源

**Android端实现**:
- 使用 Retrofit/OkHttp 进行HTTP请求
- 使用 Multipart 上传图片
- 解析JSON响应获取处理结果

**示例代码**:
```kotlin
interface MapProcessingService {
    @Multipart
    @POST("map/process")
    suspend fun processMap(
        @Part file: MultipartBody.Part,
        @Query("apply_perspective") applyPerspective: Boolean,
        @Query("extract_ocr") extractOcr: Boolean
    ): Response<MapProcessingResult>
}
```

#### 方案2: 模型部署到Android（未来优化）

**技术栈**:
- **图像处理**: OpenCV for Android
- **OCR**: PaddleOCR → ONNX Runtime / TensorFlow Lite
- **图处理**: 自定义实现

**步骤**:
1. 将PaddleOCR模型转换为ONNX格式
2. 使用ONNX Runtime Android版运行模型
3. 将图像处理逻辑移植到Android（使用OpenCV Android SDK）
4. 实现轻量级图结构处理

**优点**:
- 离线处理，无需网络
- 响应速度快
- 保护用户隐私

**缺点**:
- 实现复杂
- 需要模型转换和优化
- 增加APK体积

## 处理流程

```
1. 图像加载
   ↓
2. 透视变换（可选）
   ↓
3. OCR文本提取（可选）
   ↓
4. 图像预处理（灰度化、二值化、形态学操作）
   ↓
5. 骨架化（将线条细化为1像素）
   ↓
6. 图结构提取（节点和边）
   ↓
7. 节点分类（路口、转角、端点）
   ↓
8. OCR-节点关联和设施识别
   ↓
9. 输出结构化数据
```

## 配置参数

可在 `MapDigitizer` 初始化时配置：

- `adaptive_block_size`: 自适应阈值块大小（默认35）
- `adaptive_c`: 自适应阈值常数（默认10）
- `morph_kernel_size`: 形态学操作核大小（默认3）
- `use_ocr`: 是否启用OCR（默认True）
- `ocr_lang`: OCR语言（"ch"中文或"en"英文）
- `ocr_node_distance_threshold`: OCR与节点关联距离阈值（默认100像素）
- `corner_angle_tolerance`: 转角角度容差（默认30度）

## 使用示例

### Python直接调用

```python
from app.services.map_processor import MapDigitizer

digitizer = MapDigitizer(use_ocr=True)
result = digitizer.process_fire_map(
    image_path="demo.jpg",
    apply_perspective=True,
    extract_ocr=True
)

# 访问结果
for node in result['nodes']:
    print(f"节点{node['id']}: {node.get('node_type')}")
    if node.get('facility_type'):
        print(f"  设施类型: {node['facility_type']}")
```

### API调用

```bash
curl -X POST "http://localhost:8000/map/process" \
  -F "file=@demo.jpg" \
  -F "apply_perspective=true" \
  -F "extract_ocr=true"
```

## 测试脚本

提供了两个测试脚本：

1. **scripts/test_map_processor_direct.py**: 直接测试处理功能（不通过API）
2. **scripts/test_map_processing.py**: 通过API测试

使用方法：
```bash
# 直接测试
python scripts/test_map_processor_direct.py demo.jpg

# API测试（需要先启动服务器）
python scripts/test_map_processing.py demo.jpg
```

## 性能考虑

- **处理时间**: 取决于图片大小和复杂度，通常几秒到几十秒
- **内存占用**: 主要取决于图片分辨率
- **OCR性能**: PaddleOCR首次运行需要下载模型，后续会缓存

## 未来优化方向

1. **性能优化**:
   - 图片预处理优化
   - 并行处理
   - 缓存机制

2. **准确性提升**:
   - 更精确的转角检测
   - 更完善的设施识别规则
   - 机器学习模型优化

3. **Android端优化**:
   - 模型量化
   - 模型剪枝
   - 边缘计算

4. **功能扩展**:
   - 支持更多设施类型
   - 路径规划
   - 3D可视化

## 相关文档

- [Windows调试指南](windows_debug_guide.md)
- [技术栈路线图](tech_stack_roadmap.md)
