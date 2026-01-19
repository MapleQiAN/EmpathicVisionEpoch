# 🚀 下一步行动计划

## ✅ Sprint 1 已完成

你现在已经完成了Sprint 1的核心功能：

- ✅ 透视变换（自动检测四个角并校正）
- ✅ OCR文本提取（PaddleOCR集成）
- ✅ 骨架化算法（提取拓扑结构）
- ✅ OCR-节点关联（自动将文本绑定到节点）
- ✅ RESTful API端点（`POST /map/process`）

## 📋 立即可以做的事情

### 1. 测试现有功能

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动后端服务
cd backend
uvicorn app.main:app --reload

# 3. 在另一个终端测试API
python scripts/test_map_processing.py ../demo.jpg
```

或者直接访问 http://localhost:8000/docs 使用Swagger UI测试。

### 2. 准备测试数据

收集一些消防图图片：
- 清晰的消防疏散图
- 最好是包含房间号、楼梯标识的
- 格式：jpg/png

### 3. 验证结果

处理完图片后，检查：
- 节点数量是否合理（交叉点、端点）
- OCR是否正确识别了房间号、楼梯等
- 节点和OCR文本是否关联正确

## 🎯 Sprint 2 准备：视觉指纹（The Eyes）

### 需要准备的工具和环境

1. **SuperPoint + SuperGlue**
   ```bash
   pip install torch torchvision
   # SuperPoint需要从GitHub安装
   git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git
   ```

2. **向量数据库（可选）**
   - 简单场景：用NumPy + pickle先顶
   - 生产环境：Faiss或Milvus

3. **视频采集工具**
   - 手机录制走廊视频
   - 或使用OpenCV从摄像头采集

### Sprint 2 任务清单

- [ ] **关键帧提取**
  - 从视频中提取帧
  - 过滤模糊/过曝/纯色帧
  - 保存高质量关键帧

- [ ] **特征提取**
  - 集成SuperPoint提取特征点
  - 生成图像描述符（256/512维向量）
  - 保存特征向量和元数据

- [ ] **向量库建立**
  - 设计存储结构（图像ID -> 特征向量）
  - 实现相似度搜索接口
  - 建立索引（Faiss/Milvus或NumPy）

- [ ] **API端点**
  - `POST /vision/extract` - 提取单张图片特征
  - `POST /vision/search` - 搜索相似图片
  - `GET /vision/keyframes` - 列出所有关键帧

## 🔄 Sprint 3 准备：融合与定位

### 核心算法设计

1. **OCR锚定算法**
   ```
   视频帧 -> OCR检测"402" -> 查找地图节点"402" -> 位置校准
   ```

2. **序列对齐**
   ```
   当前帧位置 + 光流估计移动距离 -> 预测下一位置 -> 匹配关键帧
   ```

3. **闭环检测**
   ```
   识别特定视觉锚点（灭火器箱、标志牌）-> 修正位置误差
   ```

### 需要的技术栈

- 光流估计：OpenCV的`calcOpticalFlowPyrLK`
- 位置估计：卡尔曼滤波或粒子滤波
- 路径规划：NetworkX的最短路径算法

## 📱 Sprint 4 准备：导航交互

### Flutter开发环境

```bash
# 安装Flutter SDK
# https://flutter.dev/docs/get-started/install

# 创建Flutter项目
flutter create empathic_vision_app
cd empathic_vision_app
```

### 需要的功能模块

1. **摄像头模块**
   - 实时取景器（Viewfinder）
   - 视频录制
   - 帧提取

2. **API客户端**
   - HTTP请求封装
   - 图片上传
   - 实时定位查询

3. **UI界面**
   - "我在哪"按钮
   - "要去哪"搜索框
   - 关键帧显示
   - 导航指引

## 🛠️ 开发建议

### 优先级排序

1. **先完善Sprint 1**（当前）
   - 测试各种消防图
   - 优化参数（透视变换、OCR阈值）
   - 处理边界情况

2. **然后开始Sprint 2**（视觉指纹）
   - 这是定位的基础
   - 可以先不用向量数据库，用NumPy文件存储

3. **最后做Sprint 3和4**（定位和UI）
   - 依赖Sprint 2的成果
   - 可以并行开发（后端定位算法 + 前端UI）

### 开发流程建议

1. **先做后端，再做前端**
   - 在Jupyter/脚本中验证算法
   - 然后封装成API
   - 最后做Flutter前端

2. **增量开发**
   - 每个功能先做最小可用版本（MVP）
   - 测试通过后再优化

3. **数据驱动**
   - 收集真实数据（消防图、走廊视频）
   - 用真实数据测试和调优

## 📚 学习资源

- **PaddleOCR文档**: https://github.com/PaddlePaddle/PaddleOCR
- **SuperPoint论文**: https://arxiv.org/abs/1712.07629
- **FastAPI文档**: https://fastapi.tiangolo.com/
- **Flutter文档**: https://flutter.dev/docs

## 🐛 遇到问题？

1. 查看 `docs/sprint1_guide.md` 的常见问题部分
2. 检查API文档：http://localhost:8000/docs
3. 查看日志输出

---

**下一步：开始测试Sprint 1的功能，然后准备Sprint 2的开发环境！** 🚀
