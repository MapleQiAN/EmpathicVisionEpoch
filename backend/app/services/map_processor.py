from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

# PaddleOCR 是可选依赖：未安装时自动禁用 OCR 功能
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore


class NodeType(str, Enum):
    """节点类型枚举"""
    INTERSECTION = "intersection"  # 路口：连接3条或以上路径
    CORNER = "corner"  # 转角：连接2条路径且角度接近90度
    ENDPOINT = "endpoint"  # 端点：只连接1条路径
    FACILITY = "facility"  # 重要设施：通过OCR识别


class FacilityType(str, Enum):
    """重要设施类型枚举"""
    STAIRS = "stairs"  # 楼梯
    ELEVATOR = "elevator"  # 电梯
    EXIT = "exit"  # 安全出口
    FIRE_HYDRANT = "fire_hydrant"  # 消防栓
    TOILET = "toilet"  # 卫生间
    ROOM = "room"  # 房间
    UNKNOWN = "unknown"  # 未知设施


class OCRResult:
    """OCR识别结果"""
    def __init__(self, text: str, bbox: List[List[int]], confidence: float):
        self.text = text
        self.bbox = bbox  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        self.confidence = confidence
        self.center_x = int(np.mean([p[0] for p in bbox]))
        self.center_y = int(np.mean([p[1] for p in bbox]))


class MapDigitizer:
    """Digitize a fire-escape map image into a topological graph.
    
    支持Android迁移的技术栈说明：
    - 后端：Python + FastAPI（当前实现）
    - Android端：可通过REST API调用，或使用ONNX Runtime/TensorFlow Lite部署模型
    - 图像处理：OpenCV（有Android版本）
    - OCR：PaddleOCR（可转换为ONNX模型在Android上运行）
    """

    def __init__(
        self,
        adaptive_block_size: int = 35,
        adaptive_c: int = 10,
        morph_kernel_size: int = 3,
        use_ocr: bool = True,
        ocr_lang: str = "ch",  # ch: 中文, en: 英文
        ocr_node_distance_threshold: float = 100.0,  # OCR与节点关联的距离阈值（像素）
        corner_angle_tolerance: float = 30.0,  # 转角角度容差（度）
    ) -> None:
        # Tuning knobs for thresholding and denoising.
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.use_ocr = use_ocr
        self.ocr_node_distance_threshold = ocr_node_distance_threshold
        self.corner_angle_tolerance = corner_angle_tolerance
        self.ocr = None
        if use_ocr:
            try:
                if PaddleOCR is None:
                    raise ImportError("paddleocr is not installed")
                # 初始化PaddleOCR，use_angle_cls=True启用方向分类器
                self.ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False)
                print("[INFO] PaddleOCR initialized successfully")
            except Exception as e:
                print(f"[WARN] Failed to initialize PaddleOCR: {e}, OCR features disabled")
                self.use_ocr = False

    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """检测消防图的四个角点，用于透视变换。
        
        返回: 4x2的numpy数组，格式为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        如果检测失败返回None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 方法1: 使用轮廓检测找最大矩形
        # 先做边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作连接断开的边缘
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找最大的轮廓（假设是地图边界）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 近似为多边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 如果是4个点，直接返回
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            # 排序：左上、右上、右下、左下
            corners = self._sort_corners(corners)
            return corners
        
        # 如果不是4个点，尝试找最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        corners = self._sort_corners(box)
        return corners

    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """将角点排序为：左上、右上、右下、左下"""
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 按角度排序
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        # 找到最左上角的点（x+y最小）
        sums = corners.sum(axis=1)
        top_left_idx = np.argmin(sums)
        
        # 重新排列，从左上角开始
        sorted_corners = np.roll(corners, -top_left_idx, axis=0)
        
        # 确保顺序：左上、右上、右下、左下
        # 通过比较x坐标确定左右
        if sorted_corners[1][0] < sorted_corners[3][0]:
            # 交换右上和左下
            sorted_corners[[1, 3]] = sorted_corners[[3, 1]]
        
        return sorted_corners

    def perspective_transform(
        self, 
        image: np.ndarray, 
        corners: Optional[np.ndarray] = None,
        target_width: int = 2000,
        target_height: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对图像进行透视变换，拉正地图。
        
        返回: (校正后的图像, 变换矩阵)
        """
        if corners is None:
            corners = self.detect_corners(image)
        
        if corners is None:
            print("[WARN] 无法检测到四个角点，跳过透视变换")
            return image, np.eye(3, dtype=np.float32)
        
        # 目标矩形的四个角点
        dst_corners = np.array([
            [0, 0],  # 左上
            [target_width, 0],  # 右上
            [target_width, target_height],  # 右下
            [0, target_height]  # 左下
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
        
        # 应用变换
        warped = cv2.warpPerspective(image, M, (target_width, target_height))
        
        return warped, M

    def extract_text_with_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """使用PaddleOCR提取图像中的文本及其坐标。
        
        返回: OCRResult列表
        """
        if not self.use_ocr or self.ocr is None:
            return []
        
        try:
            # PaddleOCR返回格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
            results = self.ocr.ocr(image, cls=True)
            
            ocr_results = []
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        ocr_results.append(OCRResult(text, bbox, confidence))
            
            print(f"[INFO] OCR提取到 {len(ocr_results)} 个文本")
            return ocr_results
        except Exception as e:
            print(f"[ERROR] OCR处理失败: {e}")
            return []

    def process_fire_map(
        self, 
        image_path: str,
        apply_perspective: bool = True,
        extract_ocr: bool = True
    ) -> Dict[str, any]:
        """Process a fire-escape map image and return nodes + edges + OCR results.

        The output dictionary contains:
          - nodes: list of {id, x, y}
          - edges: list of {source_node_id, target_node_id, length, weight}
          - ocr_results: list of {text, center_x, center_y, bbox, confidence}
          - perspective_matrix: 透视变换矩阵（如果应用了）
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_image = image.copy()
        perspective_matrix = None
        
        # Step 0: 透视变换（如果启用）
        if apply_perspective:
            try:
                image, perspective_matrix = self.perspective_transform(image)
                print("[INFO] 透视变换完成")
            except Exception as e:
                print(f"[WARN] 透视变换失败: {e}，使用原图继续处理")

        # Step 0.5: 基于国标常见表达（绿色引导线/疏散路线）识别疏散路径
        # 注意：这不是“所有线条=通道”，而是优先抽取绿色疏散路径语义
        evacuation_routes = self._extract_evacuation_routes(image)

        # Step 1: OCR文本提取（如果启用）
        ocr_results = []
        if extract_ocr and self.use_ocr:
            ocr_results = self.extract_text_with_ocr(image)
            # 转换为字典格式便于序列化
            ocr_dicts = [
                {
                    "text": r.text,
                    "center_x": r.center_x,
                    "center_y": r.center_y,
                    "bbox": r.bbox,
                    "confidence": float(r.confidence)
                }
                for r in ocr_results
            ]
        else:
            ocr_dicts = []

        # Step 2: Preprocessing - grayscale and adaptive threshold for line isolation.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Invert threshold so black lines become white foreground for skeletonization.
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_c,
        )
        # Morphological opening removes small noise while preserving line structures.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Step 3: Skeletonization - reduce line thickness to 1px for graph tracing.
        skeleton = skeletonize(binary > 0)

        # Step 4: Graph extraction from skeleton pixels.
        graph, nodes = self._skeleton_to_graph(skeleton)

        # Step 5: 节点分类和增强（路口、转角、端点识别）
        nodes = self._classify_nodes(graph, nodes)

        # Step 6: OCR-节点关联和重要设施识别
        if ocr_results:
            nodes = self._link_ocr_to_nodes(nodes, ocr_results, graph)

        edges: List[Dict[str, int | float]] = []
        for source, target, data in graph.edges(data=True):
            length = float(data.get("length", data.get("weight", 0.0)))
            edges.append(
                {
                    "source_node_id": int(source),
                    "target_node_id": int(target),
                    "length": length,
                    "weight": length,
                }
            )

        # Step 7: 结构拓扑与疏散语义分离
        # - edges/nodes: 用于结构拓扑（墙体/线框/连接关系）
        # - evacuation_routes: 用于国标语义的疏散路线（通常为绿色引导线）

        result = {
            "nodes": nodes,
            "edges": edges,
            "ocr_results": ocr_dicts,
            "evacuation_routes": evacuation_routes,
        }
        
        if perspective_matrix is not None:
            result["perspective_matrix"] = perspective_matrix.tolist()

        return result

    def _extract_evacuation_routes(self, image_bgr: np.ndarray) -> List[Dict[str, any]]:
        """从国标消防疏散图中抽取“疏散路线/引导线”（通常为绿色）。

        说明：
        - 国标疏散图中，疏散路线常用绿色线/箭头表现（具体印刷会有差异）
        - 这里先用颜色语义做第一阶段抽取：HSV 绿区域 -> 去噪/连通 -> 骨架/轮廓 -> polyline

        Returns:
            evacuation_routes: [{ "path": [{"x":..,"y":..}, ...], "kind": "green_route"}]
        """
        try:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

            # 绿色范围（经验阈值，后续可按数据集再调）
            # Hue: 35~85 约覆盖常见绿色；S/V 下限用于排除灰/白背景
            lower = np.array([35, 60, 60], dtype=np.uint8)
            upper = np.array([85, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # 去噪/连通：先开运算去小点，再闭运算连接断裂线
            k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

            # 如果绿色像素太少，认为未检测到可靠疏散路线
            if int(cv2.countNonZero(mask)) < 200:
                return []

            # 细化：用 skeletonize 得到中心线，再提取轮廓做 polyline
            skel = skeletonize(mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            routes: List[Dict[str, any]] = []
            for cnt in contours:
                if cnt is None or len(cnt) < 30:
                    continue
                # cnt: (N,1,2) -> (N,2)
                pts = cnt.reshape(-1, 2)
                # 简化 polyline，减少点数
                eps = 2.0
                approx = cv2.approxPolyDP(pts, epsilon=eps, closed=False)
                approx_pts = approx.reshape(-1, 2)
                if len(approx_pts) < 2:
                    continue
                routes.append(
                    {
                        "kind": "green_route",
                        "path": [{"x": float(x), "y": float(y)} for x, y in approx_pts],
                    }
                )

            # 按路径长度（点数）降序，方便前端优先展示主路径
            routes.sort(key=lambda r: len(r.get("path", [])), reverse=True)
            return routes
        except Exception as e:
            print(f"[WARN] 疏散路线识别失败: {e}")
            return []

    def _skeleton_to_graph(
        self, skeleton: np.ndarray
    ) -> Tuple[nx.Graph, List[Dict[str, int]]]:
        height, width = skeleton.shape
        skeleton_points = np.argwhere(skeleton)
        if skeleton_points.size == 0:
            return nx.Graph(), []

        skeleton_set = {tuple(pt) for pt in skeleton_points}

        # 8-connected neighbor offsets for tracing pixel paths.
        neighbor_offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

        def neighbors(coord: Tuple[int, int]) -> List[Tuple[int, int]]:
            y, x = coord
            neighbor_list: List[Tuple[int, int]] = []
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and (ny, nx) in skeleton_set:
                    neighbor_list.append((ny, nx))
            return neighbor_list

        neighbor_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for coord in skeleton_set:
            neighbor_map[coord] = neighbors(coord)

        # Node definition: endpoints (1 neighbor) or junctions (>2 neighbors).
        node_coords: List[Tuple[int, int]] = []
        for coord, neigh in neighbor_map.items():
            count = len(neigh)
            if count <= 1 or count > 2:
                node_coords.append(coord)

        graph = nx.Graph()
        nodes: List[Dict[str, int]] = []
        node_index: Dict[Tuple[int, int], int] = {}
        for node_id, (y, x) in enumerate(node_coords):
            node_index[(y, x)] = node_id
            nodes.append({"id": node_id, "x": int(x), "y": int(y)})
            graph.add_node(node_id, x=int(x), y=int(y))

        def segment_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int, int, int]:
            return (*a, *b) if a <= b else (*b, *a)

        visited_segments: set[Tuple[int, int, int, int]] = set()

        # Trace edges by walking from node to node along skeleton pixels.
        for start_coord, start_id in node_index.items():
            for neighbor in neighbor_map.get(start_coord, []):
                if segment_key(start_coord, neighbor) in visited_segments:
                    continue

                prev = start_coord
                curr = neighbor
                length = 0.0

                while True:
                    visited_segments.add(segment_key(prev, curr))
                    length += float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))

                    if curr in node_index and curr != start_coord:
                        end_id = node_index[curr]
                        if not graph.has_edge(start_id, end_id):
                            graph.add_edge(start_id, end_id, length=length, weight=length)
                        break

                    next_candidates = [n for n in neighbor_map.get(curr, []) if n != prev]
                    if not next_candidates:
                        break

                    next_pixel = None
                    for candidate in next_candidates:
                        if segment_key(curr, candidate) not in visited_segments:
                            next_pixel = candidate
                            break

                    if next_pixel is None:
                        break

                    prev, curr = curr, next_pixel

        return graph, nodes

    def _classify_nodes(
        self, 
        graph: nx.Graph, 
        nodes: List[Dict[str, int]]
    ) -> List[Dict[str, any]]:
        """对节点进行分类：识别路口、转角、端点
        
        Args:
            graph: NetworkX图对象
            nodes: 原始节点列表
            
        Returns:
            增强后的节点列表，包含node_type、degree、angle等字段
        """
        enhanced_nodes = []
        
        for node in nodes:
            node_id = node["id"]
            degree = graph.degree(node_id)
            
            # 初始化节点信息
            enhanced_node = {
                **node,
                "node_type": NodeType.ENDPOINT.value,
                "degree": int(degree),
                "confidence": 0.8,  # 默认置信度
            }
            
            # 分类逻辑
            if degree == 1:
                # 端点：只连接1条路径
                enhanced_node["node_type"] = NodeType.ENDPOINT.value
                enhanced_node["confidence"] = 0.9
                
            elif degree >= 3:
                # 路口：连接3条或以上路径
                enhanced_node["node_type"] = NodeType.INTERSECTION.value
                enhanced_node["confidence"] = 0.85
                
            elif degree == 2:
                # 可能是转角或普通路径节点
                # 计算两条边的角度来判断是否为转角
                angle = self._calculate_node_angle(graph, node_id, node)
                if angle is not None:
                    # 检查角度是否接近90度（允许容差）
                    angle_diff = abs(angle - 90.0)
                    if angle_diff <= self.corner_angle_tolerance:
                        enhanced_node["node_type"] = NodeType.CORNER.value
                        enhanced_node["angle"] = float(angle)
                        enhanced_node["confidence"] = 0.8
                    else:
                        # 普通路径节点，保持为endpoint类型（或可以添加新类型）
                        enhanced_node["node_type"] = NodeType.ENDPOINT.value
                        enhanced_node["confidence"] = 0.7
                else:
                    enhanced_node["node_type"] = NodeType.ENDPOINT.value
                    enhanced_node["confidence"] = 0.7
            
            enhanced_nodes.append(enhanced_node)
        
        return enhanced_nodes

    def _calculate_node_angle(
        self, 
        graph: nx.Graph, 
        node_id: int, 
        node: Dict[str, int]
    ) -> Optional[float]:
        """计算节点的角度（用于判断是否为转角）
        
        对于度数为2的节点，计算两条边的夹角
        
        Args:
            graph: NetworkX图对象
            node_id: 节点ID
            node: 节点信息（包含x, y坐标）
            
        Returns:
            角度（度），如果无法计算返回None
        """
        neighbors = list(graph.neighbors(node_id))
        if len(neighbors) != 2:
            return None
        
        # 获取两个邻居节点的坐标
        n1_id, n2_id = neighbors[0], neighbors[1]
        n1_data = graph.nodes[n1_id]
        n2_data = graph.nodes[n2_id]
        
        # 计算两个向量
        v1 = np.array([n1_data["x"] - node["x"], n1_data["y"] - node["y"]])
        v2 = np.array([n2_data["x"] - node["x"], n2_data["y"] - node["y"]])
        
        # 计算向量长度
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 == 0 or len2 == 0:
            return None
        
        # 计算夹角（使用点积）
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        # 限制在[-1, 1]范围内，避免数值误差
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def _link_ocr_to_nodes(
        self,
        nodes: List[Dict[str, any]],
        ocr_results: List[OCRResult],
        graph: nx.Graph
    ) -> List[Dict[str, any]]:
        """将OCR结果关联到最近的节点，并识别重要设施
        
        Args:
            nodes: 节点列表
            ocr_results: OCR识别结果列表
            graph: NetworkX图对象（用于获取节点度数）
            
        Returns:
            增强后的节点列表，包含facility_type、ocr_text等字段
        """
        # 为每个节点初始化设施相关字段
        for node in nodes:
            node.setdefault("facility_type", None)
            node.setdefault("ocr_text", None)
            node.setdefault("ocr_confidence", None)
        
        # 对每个OCR结果，找到最近的节点
        for ocr in ocr_results:
            ocr_x, ocr_y = ocr.center_x, ocr.center_y
            text = ocr.text.strip()
            
            # 找到最近的节点
            min_dist = float('inf')
            nearest_node_idx = None
            
            for idx, node in enumerate(nodes):
                dx = node["x"] - ocr_x
                dy = node["y"] - ocr_y
                dist = np.sqrt(dx * dx + dy * dy)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_node_idx = idx
            
            # 如果距离足够近，关联OCR结果
            if nearest_node_idx is not None and min_dist < self.ocr_node_distance_threshold:
                node = nodes[nearest_node_idx]
                
                # 识别设施类型
                facility_type = self._identify_facility_type(text)
                
                # 如果识别到重要设施，更新节点信息
                if facility_type != FacilityType.UNKNOWN:
                    node["node_type"] = NodeType.FACILITY.value
                    node["facility_type"] = facility_type.value
                    node["ocr_text"] = text
                    node["ocr_confidence"] = float(ocr.confidence)
                    # 提高置信度（因为OCR提供了额外信息）
                    node["confidence"] = min(0.95, node.get("confidence", 0.8) + 0.1)
                elif node.get("facility_type") is None:
                    # 即使不是已知设施，也记录OCR文本（可能是房间号等）
                    node["ocr_text"] = text
                    node["ocr_confidence"] = float(ocr.confidence)
        
        return nodes

    def _identify_facility_type(self, text: str) -> FacilityType:
        """根据OCR文本识别设施类型
        
        Args:
            text: OCR识别的文本
            
        Returns:
            设施类型枚举值
        """
        text_lower = text.lower().strip()
        
        # 楼梯相关关键词
        stairs_keywords = ["楼梯", "stair", "楼梯间", "安全楼梯", "疏散楼梯"]
        if any(keyword in text_lower for keyword in stairs_keywords):
            return FacilityType.STAIRS
        
        # 电梯相关关键词
        elevator_keywords = ["电梯", "elevator", "lift", "升降机"]
        if any(keyword in text_lower for keyword in elevator_keywords):
            return FacilityType.ELEVATOR
        
        # 安全出口相关关键词
        exit_keywords = ["安全出口", "exit", "出口", "emergency exit", "疏散出口"]
        if any(keyword in text_lower for keyword in exit_keywords):
            return FacilityType.EXIT
        
        # 消防栓相关关键词
        fire_hydrant_keywords = ["消防栓", "fire hydrant", "消火栓", "消防", "灭火器"]
        if any(keyword in text_lower for keyword in fire_hydrant_keywords):
            return FacilityType.FIRE_HYDRANT
        
        # 卫生间相关关键词
        toilet_keywords = ["厕所", "toilet", "洗手间", "wc", "卫生间", "盥洗室"]
        if any(keyword in text_lower for keyword in toilet_keywords):
            return FacilityType.TOILET
        
        # 房间号（通常是数字或数字+字母）
        if text.strip().replace("-", "").replace(" ", "").isdigit():
            return FacilityType.ROOM
        
        # 包含数字的房间标识（如"402", "A101"等）
        if any(char.isdigit() for char in text) and len(text.strip()) <= 10:
            # 可能是房间号
            return FacilityType.ROOM
        
        return FacilityType.UNKNOWN

    def _identify_corridors(
        self,
        graph: nx.Graph,
        nodes: List[Dict[str, any]],
        skeleton: np.ndarray,
        image_shape: Tuple[int, int] | Tuple[int, int, int]
    ) -> List[Dict[str, any]]:
        """识别走廊可行走区域
        
        策略：
        1. 沿着骨架图的边，识别走廊路径
        2. 对于每条边，生成走廊中心线路径点
        3. 可以扩展为识别走廊宽度（未来改进）
        
        Args:
            graph: NetworkX图对象
            nodes: 节点列表
            skeleton: 骨架图像
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            走廊区域列表，每个走廊包含path（路径点列表）
        """
        corridors = []
        
        # 为每个节点创建坐标映射
        node_coords = {node["id"]: (node["x"], node["y"]) for node in nodes}
        
        # 遍历所有边，生成走廊路径
        for source_id, target_id, data in graph.edges(data=True):
            source_coord = node_coords[source_id]
            target_coord = node_coords[target_id]
            
            # 生成路径点（沿着边的方向）
            path_points = self._generate_corridor_path(
                source_coord, 
                target_coord, 
                skeleton,
                image_shape
            )
            
            if path_points:
                corridors.append({
                    "source_node_id": int(source_id),
                    "target_node_id": int(target_id),
                    "path": [{"x": float(x), "y": float(y)} for x, y in path_points],
                    "length": float(data.get("length", 0.0))
                })
        
        return corridors

    def _generate_corridor_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        skeleton: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """生成走廊路径点
        
        沿着骨架图从起点到终点，生成路径点列表
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            skeleton: 骨架图像
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            路径点列表 [(x, y), ...]
        """
        # 兼容传入 (H, W) 或 (H, W, C)
        if len(image_shape) >= 2:
            height, width = int(image_shape[0]), int(image_shape[1])
        else:
            raise ValueError(f"Invalid image_shape: {image_shape}")
        path_points = []
        
        # 使用A*算法或简单直线插值
        # 简化版本：使用直线插值，然后找到骨架上的最近点
        sx, sy = start
        ex, ey = end
        
        # 计算距离
        distance = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        
        # 如果距离太短，直接返回起点和终点
        if distance < 10:
            return [(sx, sy), (ex, ey)]
        
        # 生成插值点
        num_points = max(2, int(distance / 5))  # 每5像素一个点
        for i in range(num_points + 1):
            t = i / num_points
            x = int(sx + t * (ex - sx))
            y = int(sy + t * (ey - sy))
            
            # 确保坐标在图像范围内
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # 如果点在骨架上，添加到路径
            if skeleton[y, x]:
                path_points.append((x, y))
            else:
                # 如果不在骨架上，找最近的骨架点
                nearest = self._find_nearest_skeleton_point(x, y, skeleton, radius=10)
                if nearest:
                    path_points.append(nearest)
        
        # 确保包含起点和终点
        if not path_points or path_points[0] != (sx, sy):
            path_points.insert(0, (sx, sy))
        if not path_points or path_points[-1] != (ex, ey):
            path_points.append((ex, ey))
        
        # 去重并保持顺序
        seen = set()
        unique_path = []
        for point in path_points:
            if point not in seen:
                seen.add(point)
                unique_path.append(point)
        
        return unique_path

    def _find_nearest_skeleton_point(
        self,
        x: int,
        y: int,
        skeleton: np.ndarray,
        radius: int = 10
    ) -> Optional[Tuple[int, int]]:
        """找到最近的骨架点
        
        Args:
            x, y: 查询点坐标
            skeleton: 骨架图像
            radius: 搜索半径
            
        Returns:
            最近的骨架点坐标，如果找不到返回None
        """
        height, width = skeleton.shape
        
        min_dist = float('inf')
        nearest = None
        
        # 在半径范围内搜索
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < height and 0 <= nx < width:
                    if skeleton[ny, nx]:
                        dist = dx * dx + dy * dy
                        if dist < min_dist:
                            min_dist = dist
                            nearest = (nx, ny)
        
        return nearest
