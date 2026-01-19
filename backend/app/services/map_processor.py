from __future__ import annotations

import math
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label, regionprops

# PaddleOCR 是必选依赖：系统强制使用 OCR
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
        force_ocr: bool = False,
        ocr_lang: str = "ch",  # ch: 中文, en: 英文
        ocr_node_distance_threshold: float = 100.0,  # OCR与节点关联的距离阈值（像素）
        corner_angle_tolerance: float = 30.0,  # 转角角度容差（度）
        debug_dir: str = "debug",
        save_debug: bool = True,
    ) -> None:
        # Tuning knobs for thresholding and denoising.
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        # 系统强制使用 OCR：忽略外部传入的 use_ocr/force_ocr
        self.use_ocr = True
        self.force_ocr = True
        self.ocr_node_distance_threshold = ocr_node_distance_threshold
        self.corner_angle_tolerance = corner_angle_tolerance
        self.debug_dir = debug_dir
        self.save_debug = save_debug
        self.ocr = None
        # 默认颜色阈值（当图例缺失时 fallback）
        self.default_green_range = (np.array([35, 40, 40], dtype=np.uint8), np.array([90, 255, 255], dtype=np.uint8))
        self.default_red_range = (np.array([0, 60, 60], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8))
        # corridor 提取参数
        self.wall_close_kernel = 3
        self.free_space_min_area_ratio = 0.0008
        self.corridor_min_aspect = 1.6
        self.corridor_min_slim_ratio = 0.25
        self.seed_dilate_iter = 3
        self.max_rotation_correction = 8.0  # degrees
        # OCR：必选依赖。不允许降级关闭。
        try:
            if PaddleOCR is None:
                raise ImportError("paddleocr not available")
            # PaddleOCR 参数在不同版本会变化；这里使用最小参数集以保证兼容性
            self.ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang)
            print("[INFO] PaddleOCR initialized successfully (OCR required)")
        except Exception as e:
            raise RuntimeError("系统已强制启用OCR，但 PaddleOCR 初始化失败") from e

    # ---------------------------
    # 工具与调试辅助
    # ---------------------------
    def _prepare_debug_dir(self, image_path: str) -> str:
        """按图片名创建 debug 子目录"""
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(self.debug_dir, base)
        if self.save_debug:
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_debug(self, out_dir: str, name: str, image: np.ndarray) -> None:
        """保存调试图；失败时不抛出"""
        if not self.save_debug:
            return
        try:
            path = os.path.join(out_dir, f"{name}.png")
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"[WARN] 保存调试图失败 {name}: {e}")

    def _draw_overlay(self, base: np.ndarray, lines: List[List[Tuple[int, int]]], color=(0, 255, 0)) -> np.ndarray:
        vis = base.copy()
        for pts in lines:
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], color, 2, lineType=cv2.LINE_AA)
        return vis

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
        # 系统强制 OCR：这里不允许静默返回空
        if self.ocr is None:
            raise RuntimeError("OCR 未初始化，但系统要求必须使用 OCR")
        
        try:
            # PaddleOCR 各版本接口差异较大：
            # - 老版本: ocr(img, cls=True/False)
            # - 新版本: ocr(img) 内部走 predict()，不再接受 cls 参数
            try:
                results = self.ocr.ocr(image)
            except TypeError:
                # 兼容旧版本签名
                results = self.ocr.ocr(image, cls=True)
            
            ocr_results = []
            # 兼容常见返回结构：
            # 1) results = [ [ [bbox, (text, conf)], ... ] ]  （老版本常见）
            # 2) results = [ [bbox, (text, conf)], ... ]      （部分版本直接返回单层）
            if results:
                lines = results[0] if isinstance(results, list) and len(results) == 1 and isinstance(results[0], list) else results
                for line in lines or []:
                    if not line:
                        continue
                    bbox, (text, confidence) = line
                    ocr_results.append(OCRResult(text, bbox, float(confidence)))
            
            print(f"[INFO] OCR提取到 {len(ocr_results)} 个文本")
            return ocr_results
        except Exception as e:
            raise RuntimeError(f"OCR处理失败: {e}") from e

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
        debug_dir = self._prepare_debug_dir(image_path)
        perspective_matrix = np.eye(3, dtype=np.float32)
        board_mask = None

        # Step 0: 透视+板面外框+去高光/螺丝干扰
        if apply_perspective:
            try:
                image, perspective_matrix, board_mask = self._robust_perspective_transform(image)
                print("[INFO] 透视矫正完成")
            except Exception as e:
                print(f"[WARN] 透视矫正失败，使用原图: {e}")
        self._save_debug(debug_dir, "rectified", image)

        glare_clean, glare_mask = self._suppress_glare(image)
        screw_clean, screw_mask = self._mask_screws(glare_clean)
        self._save_debug(debug_dir, "glare_mask", glare_mask)
        self._save_debug(debug_dir, "screw_mask", screw_mask)

        # Step 0.5: 图例驱动颜色/模板自校准
        legend_crop, legend_bbox, legend_missing = self._extract_legend_region(image)
        palettes = self._sample_palette_from_legend(legend_crop)
        palette_vis = self._visualize_palette(legend_crop, palettes, legend_missing)
        self._save_debug(debug_dir, "legend_palette", palette_vis)
        templates = self._extract_icon_templates(legend_crop, palettes)

        # Step 1: 绿色疏散箭头/路线不再输出（避免与“走廊”混淆）
        evacuation_routes: List[Dict[str, any]] = []
        seed_mask, seed_meta = self._extract_corridor_seed(image, palettes)

        # Step 2: OCR（增强小字、纠错）——可选
        ocr_results: List[OCRResult] = []
        if extract_ocr and self.use_ocr:
            ocr_input = self._prepare_ocr_image(glare_clean)
            ocr_results = self.extract_text_with_ocr(ocr_input)
            ocr_results = self._normalize_ocr_results(ocr_results)
            ocr_dicts = [
                {
                    "text": r.text,
                    "center_x": r.center_x,
                    "center_y": r.center_y,
                    "bbox": r.bbox,
                    "confidence": float(r.confidence),
                }
                for r in ocr_results
            ]
        else:
            ocr_dicts = []

        # Step 3: 结构层（墙线）+ 自适应旋转校正
        wall_mask = self._extract_wall_mask(screw_clean, screw_mask, glare_mask)
        # 去除文字/门牌区域，避免它们在结构线/节点中出现
        text_mask = self._mask_text_regions(screw_clean, ocr_results)
        if int(cv2.countNonZero(text_mask)) > 0:
            wall_mask[text_mask > 0] = 0
            self._save_debug(debug_dir, "text_mask", text_mask)
        wall_mask, rot_mat = self._deskew_mask(wall_mask)
        if rot_mat is not None:
            screw_clean = cv2.warpAffine(screw_clean, rot_mat, (wall_mask.shape[1], wall_mask.shape[0]))
            glare_mask = cv2.warpAffine(glare_mask, rot_mat, (wall_mask.shape[1], wall_mask.shape[0]))
            seed_mask = cv2.warpAffine(seed_mask, rot_mat, (wall_mask.shape[1], wall_mask.shape[0]))
            if board_mask is not None:
                board_mask = cv2.warpAffine(board_mask, rot_mat, (wall_mask.shape[1], wall_mask.shape[0]))

        self._save_debug(debug_dir, "wall_mask", wall_mask)

        # Step 4: 空间分割 -> free_space -> corridor_area（覆盖所有走廊）
        free_space = self._compute_free_space(wall_mask)
        corridor_area, corridor_info = self._build_corridor_area(free_space, seed_mask)
        self._save_debug(debug_dir, "free_space", (free_space * 255).astype(np.uint8))
        self._save_debug(debug_dir, "seed_mask", (seed_mask * 255).astype(np.uint8))
        self._save_debug(debug_dir, "corridor_area", (corridor_area * 255).astype(np.uint8))

        # Step 5: skeleton/中心线
        skeleton = skeletonize(corridor_area > 0)
        centerlines, skeleton_overlay = self._corridor_centerlines(skeleton, corridor_area)
        self._save_debug(debug_dir, "skeleton", (skeleton.astype(np.uint8) * 255))
        if skeleton_overlay is not None:
            self._save_debug(debug_dir, "centerlines_overlay", skeleton_overlay)

        # Step 6: 前端可视化用 nodes/edges：仅用“走廊骨架”生成，避免文字/门牌/疏散箭头产生点线
        corridor_skel_for_graph = self._prune_skeleton_spurs(skeleton.astype(np.uint8), max_len=18)
        graph, nodes = self._skeleton_to_graph(corridor_skel_for_graph > 0)
        nodes = self._classify_nodes(graph, nodes)
        # 用户需求：门牌号/文字不应被标成点，因此不将 OCR 结果绑定到 nodes（避免生成 FACILITY 点）

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

        # Step 7: 按需求只标注走廊，房间/设施等上层语义暂不输出（避免额外点线干扰）
        rooms: List[Dict[str, any]] = []
        facilities: List[Dict[str, any]] = []

        result = {
            "nodes": nodes,
            "edges": edges,
            "ocr_results": ocr_dicts,
            "evacuation_routes": evacuation_routes,
            "corridors": {
                "centerlines": centerlines,
                "seed_used": seed_meta,
                "corridor_info": corridor_info,
            },
            "rooms": rooms,
            "facilities": facilities,
            "legend_missing": legend_missing,
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
                # 走廊可视化中，“墙角/转角”不应被标成路口（红点）。
                # 按需求：转角统一按端点展示（避免产生大量转角/路口点污染）
                angle = self._calculate_node_angle(graph, node_id, node)
                if angle is not None:
                    enhanced_node["angle"] = float(angle)
                enhanced_node["node_type"] = NodeType.ENDPOINT.value
                enhanced_node["confidence"] = 0.75
            
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

    # ---------------------------
    # 新增稳健处理与走廊/房间/设施辅助函数
    # ---------------------------
    def _robust_perspective_transform(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """板面外框检测 + 透视矫正，返回矫正图、矩阵、板面mask"""
        h, w = image.shape[:2]
        scale = 1200.0 / max(h, w)
        if scale < 0.5:
            scale = 0.5
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, k, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, np.eye(3, dtype=np.float32), np.ones((h, w), dtype=np.uint8) * 255
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.015 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) < 4:
            rect = cv2.minAreaRect(largest)
            approx = cv2.boxPoints(rect)
        corners = approx.reshape(-1, 2).astype(np.float32)
        corners = corners / scale
        if len(corners) > 4:
            # 取凸包再取四个极点
            hull = cv2.convexHull(corners)
            corners = hull.reshape(-1, 2)
        if corners.shape[0] != 4:
            # 兜底使用 minAreaRect
            rect = cv2.minAreaRect(corners.astype(np.float32))
            corners = cv2.boxPoints(rect)
        corners = self._sort_corners(corners)
        warped, M = self.perspective_transform(image, corners.astype(np.float32))
        board_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(board_mask, [corners.astype(np.int32)], 255)
        return warped, M, board_mask

    def _suppress_glare(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测高亮低饱和区域并做轻度修复"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        glare_mask = np.zeros_like(v)
        glare_mask[(v > 225) & (s < 60)] = 255
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(lab)
        glare_mask[(L > 240) & (np.abs(a - 128) < 6) & (np.abs(b - 128) < 6)] = 255
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if int(cv2.countNonZero(glare_mask)) == 0:
            return image, glare_mask
        inpainted = cv2.inpaint(image, glare_mask, 5, cv2.INPAINT_TELEA)
        return inpainted, glare_mask

    def _mask_screws(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测螺丝/圆形暗点并遮挡"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        mask = np.zeros_like(gray)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=30,
            param1=80,
            param2=18,
            minRadius=3,
            maxRadius=18,
        )
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            for x, y, r in circles:
                cv2.circle(mask, (x, y), int(r * 1.2), 255, -1)
        if int(cv2.countNonZero(mask)) == 0:
            return image, mask
        cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return cleaned, mask

    def _extract_legend_region(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int], bool]:
        """默认取底部 20% 作为图例区，可根据文本密度微调"""
        h, w = image.shape[:2]
        y0 = int(h * 0.75)
        legend_band = image[y0:, :]
        gray = cv2.cvtColor(legend_band, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad = cv2.convertScaleAbs(grad)
        projection = np.sum(grad, axis=1)
        if projection.max() > 0:
            top_rel = np.argmax(projection) / max(1, projection.size)
            # 如果文字集中在更靠上的地方，上移窗口
            if top_rel < 0.3:
                y0 = max(0, int(h * 0.65))
                legend_band = image[y0:, :]
        legend_missing = False
        if np.std(legend_band) < 8:
            legend_missing = True
        return legend_band, (0, y0, w, h), legend_missing

    def _sample_palette_from_legend(self, legend_crop: Optional[np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """从图例自动估计绿色/红色 HSV 范围，找不到时回退默认"""
        if legend_crop is None or legend_crop.size == 0:
            return {"green": self.default_green_range, "red": self.default_red_range}
        hsv = cv2.cvtColor(legend_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask_sat = s > 50
        green_mask = (h > 35) & (h < 90) & mask_sat
        red_mask = ((h < 15) | (h > 160)) & mask_sat

        def _range_from_mask(mask):
            if int(mask.sum()) == 0:
                return None
            hs = h[mask]
            ss = s[mask]
            vs = v[mask]
            low = np.array([
                np.clip(np.percentile(hs, 5), 0, 179),
                np.clip(np.percentile(ss, 5), 10, 255),
                np.clip(np.percentile(vs, 5), 10, 255),
            ], dtype=np.uint8)
            high = np.array([
                np.clip(np.percentile(hs, 95), 0, 179),
                np.clip(np.percentile(ss, 95), 40, 255),
                np.clip(np.percentile(vs, 95), 40, 255),
            ], dtype=np.uint8)
            return (low, high)

        green_range = _range_from_mask(green_mask) or self.default_green_range
        red_range = _range_from_mask(red_mask) or self.default_red_range
        return {"green": green_range, "red": red_range}

    def _visualize_palette(self, legend_crop: Optional[np.ndarray], palettes: Dict[str, Tuple[np.ndarray, np.ndarray]], missing: bool) -> np.ndarray:
        """生成简单调色板可视化"""
        canvas = np.ones((120, 240, 3), dtype=np.uint8) * 255
        colors = {
            "green": cv2.cvtColor(
                np.uint8([[palettes["green"][0], palettes["green"][1]]]), cv2.COLOR_HSV2BGR
            ),
            "red": cv2.cvtColor(
                np.uint8([[palettes["red"][0], palettes["red"][1]]]), cv2.COLOR_HSV2BGR
            ),
        }
        cv2.rectangle(canvas, (10, 20), (110, 100), tuple(int(x) for x in colors["green"][0, 1]), -1)
        cv2.rectangle(canvas, (130, 20), (230, 100), tuple(int(x) for x in colors["red"][0, 1]), -1)
        text = "legend_missing" if missing else "legend_ok"
        cv2.putText(canvas, text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if legend_crop is not None and legend_crop.size > 0:
            small = cv2.resize(legend_crop, (canvas.shape[1], canvas.shape[0]))
            canvas = cv2.hconcat([canvas, small])
        return canvas

    def _extract_icon_templates(self, legend_crop: Optional[np.ndarray], palettes: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
        """从图例截取安全出口/消防设施模板，失败返回空 dict"""
        templates: Dict[str, np.ndarray] = {}
        if legend_crop is None or legend_crop.size == 0:
            return templates
        hsv = cv2.cvtColor(legend_crop, cv2.COLOR_BGR2HSV)
        for key, rng in [("exit", palettes["green"]), ("fire", palettes["red"])]:
            mask = cv2.inRange(hsv, rng[0], rng[1])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) < 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            patch = legend_crop[y : y + h, x : x + w]
            templates[key] = patch
        return templates

    def _extract_corridor_seed(self, image_bgr: np.ndarray, palettes: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, Dict[str, any]]:
        """用绿色弱监督生成 seed，输出二值mask和元信息"""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower, upper = palettes.get("green", self.default_green_range)
        mask = cv2.inRange(hsv, lower, upper)
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)
        meta = {"seed_pixels": int(cv2.countNonZero(mask))}
        return (mask > 0).astype(np.uint8), meta

    def _prepare_ocr_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """针对小字做增强与放大"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # unsharp mask
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        # 放大到 1.5x
        h, w = sharp.shape
        self._ocr_scale = 1.5
        sharp = cv2.resize(sharp, (int(w * self._ocr_scale), int(h * self._ocr_scale)))
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    def _normalize_room_text(self, text: str) -> Optional[str]:
        """房号纠错：O->0, I/l->1, S->5"""
        if not text:
            return None
        t = text.strip().replace(" ", "").replace("-", "")
        if len(t) < 2 or len(t) > 6:
            return None
        table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "|": "1"})
        t = t.translate(table)
        if not any(ch.isdigit() for ch in t):
            return None
        return t

    def _normalize_ocr_results(self, ocr_results: List[OCRResult]) -> List[OCRResult]:
        """对 OCR 文本做简单纠错，主要作用于房号"""
        for r in ocr_results:
            fixed = self._normalize_room_text(r.text)
            if fixed:
                r.text = fixed
        # 如果 OCR 输入做了放大，这里把坐标缩回当前工作坐标系
        scale = getattr(self, "_ocr_scale", 1.0)
        if scale and abs(scale - 1.0) > 1e-3:
            inv = 1.0 / float(scale)
            for r in ocr_results:
                r.center_x = int(r.center_x * inv)
                r.center_y = int(r.center_y * inv)
                try:
                    r.bbox = [[int(p[0] * inv), int(p[1] * inv)] for p in r.bbox]
                except Exception:
                    pass
        return ocr_results

    def _extract_wall_mask(self, image_bgr: np.ndarray, screw_mask: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
        """生成墙线结构 mask"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # 遮掉螺丝/高光
        gray = gray.copy()
        gray[screw_mask > 0] = np.median(gray)
        gray[glare_mask > 0] = np.median(gray)
        blur = cv2.bilateralFilter(gray, 5, 15, 15)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_c,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def _mask_text_regions(self, image_bgr: np.ndarray, ocr_results: List[OCRResult]) -> np.ndarray:
        """生成文本区域mask（OCR框优先，其次MSER兜底），用于从结构线里扣掉文字干扰"""
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # 1) OCR bbox（更准）
        for r in ocr_results or []:
            try:
                pts = np.array(r.bbox, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)
            except Exception:
                continue
        if int(cv2.countNonZero(mask)) > 0:
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
            return mask
        # 2) MSER 兜底（无OCR/失败时）
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        # OpenCV 版本差异：有的版本不支持构造函数关键字参数
        try:
            mser.setMinArea(60)
            mser.setMaxArea(5000)
        except Exception:
            pass
        regions, _ = mser.detectRegions(gray)
        for reg in regions:
            x, y, rw, rh = cv2.boundingRect(reg.reshape(-1, 1, 2))
            if rw < 8 or rh < 8:
                continue
            aspect = rw / float(rh + 1e-5)
            if 0.2 < aspect < 8.0 and (rw * rh) < (h * w * 0.02):
                cv2.rectangle(mask, (x, y), (x + rw, y + rh), 255, -1)
        if int(cv2.countNonZero(mask)) > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        return mask

    def _deskew_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """估计曼哈顿主方向，小角度旋转校正"""
        lines = cv2.HoughLines(mask, 1, np.pi / 180.0, threshold=200)
        if lines is None or len(lines) == 0:
            return mask, None
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) % 180
            if angle > 90:
                angle -= 180
            angles.append(angle)
        if not angles:
            return mask, None
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5 or abs(median_angle) > self.max_rotation_correction:
            return mask, None
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return rotated, rot_mat

    def _compute_free_space(self, wall_mask: np.ndarray) -> np.ndarray:
        """用墙体反相得到可行空域"""
        inv = (wall_mask == 0)
        inv = remove_small_objects(inv, min_size=int(inv.size * self.free_space_min_area_ratio))
        free_space = inv.astype(np.uint8)
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return free_space

    def _build_corridor_area(self, free_space: np.ndarray, seed_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, any]]:
        """两阶段：seed 引导 + fallback slender 区域；输出尽量覆盖所有走廊连通域的并集"""
        info: Dict[str, any] = {}
        fs = (free_space > 0).astype(np.uint8)
        if seed_mask is None:
            seed_mask = np.zeros_like(fs)
        seed_pixels = int(cv2.countNonZero(seed_mask))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fs, 8)
        info["used_seed"] = bool(seed_pixels > 30)

        # 1) seed 引导：找到 seed 覆盖到的连通域，做轻度膨胀但不跨越墙线
        corridor = np.zeros_like(fs)
        seed_cc_ids: set[int] = set()
        if info["used_seed"] and num > 1:
            ys, xs = np.where(seed_mask > 0)
            for y, x in zip(ys.tolist(), xs.tolist()):
                cc = int(labels[y, x])
                if cc > 0:
                    seed_cc_ids.add(cc)
            for cc in seed_cc_ids:
                corridor[labels == cc] = 1
            # 在该连通域内部做“补连通”膨胀
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            growth = corridor.copy()
            for _ in range(4):
                growth = cv2.dilate(growth, k, iterations=1)
                growth = cv2.bitwise_and(growth, fs)
            corridor = growth

        # 2) fallback：对所有 free_space 连通域按“细长/走廊形状”打分，取 Top-K 并集（覆盖多走廊）
        scored: List[Tuple[float, int]] = []
        if num > 1:
            for i in range(1, num):
                area = float(stats[i, cv2.CC_STAT_AREA])
                w = float(stats[i, cv2.CC_STAT_WIDTH])
                h = float(stats[i, cv2.CC_STAT_HEIGHT])
                if w <= 1 or h <= 1:
                    continue
                aspect = max(w, h) / max(1.0, min(w, h))
                fill = area / max(1.0, w * h)
                # slender 越小越像走廊（细长/带状）；aspect 越大越像走廊
                slender = 1.0 - fill
                score = (aspect - 1.0) * 0.7 + slender * 0.9 + math.log1p(area) * 0.05
                # seed 命中的连通域额外加分
                if i in seed_cc_ids:
                    score += 2.0
                scored.append((score, i))
            scored.sort(reverse=True, key=lambda x: x[0])
            k_keep = min(6, max(2, int(len(scored) * 0.25)))
            keep_ids = {i for _, i in scored[:k_keep] if _ > 0.2}
            # 兜底：至少保留 seed 命中的
            keep_ids |= seed_cc_ids
            for i in keep_ids:
                corridor[labels == i] = 1
        else:
            corridor = fs

        corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        corridor = cv2.morphologyEx(corridor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return corridor.astype(np.uint8), info

    def _prune_skeleton_spurs(self, skel_u8: np.ndarray, max_len: int = 18) -> np.ndarray:
        """剪掉骨架上的短毛刺，减少角点附近误判为路口（degree>=3）"""
        sk = (skel_u8 > 0).astype(np.uint8)
        if sk.size == 0 or int(sk.sum()) == 0:
            return sk

        def neighbor_count(y: int, x: int) -> int:
            y0 = max(0, y - 1)
            y1 = min(sk.shape[0], y + 2)
            x0 = max(0, x - 1)
            x1 = min(sk.shape[1], x + 2)
            return int(np.sum(sk[y0:y1, x0:x1])) - int(sk[y, x])

        # 迭代：从端点开始沿着 1-2 邻居链路走 max_len，若遇到分叉则删除该链
        removed_any = True
        it = 0
        while removed_any and it < 6:
            removed_any = False
            it += 1
            ys, xs = np.where(sk > 0)
            endpoints = [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist()) if neighbor_count(int(y), int(x)) <= 1]
            for ey, ex in endpoints:
                path = [(ey, ex)]
                py, px = ey, ex
                cy, cx = ey, ex
                for _ in range(max_len):
                    # 找下一步
                    neigh = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < sk.shape[0] and 0 <= nx < sk.shape[1] and sk[ny, nx] > 0:
                                if (ny, nx) != (py, px):
                                    neigh.append((ny, nx))
                    if len(neigh) == 0:
                        break
                    if len(neigh) > 1:
                        # 分叉点：不剪主干，仅剪这条短端点链（如果链很短）
                        break
                    ny, nx = neigh[0]
                    path.append((ny, nx))
                    py, px = cy, cx
                    cy, cx = ny, nx
                    # 到达“非细链”位置（邻居>=3）就停止
                    if neighbor_count(cy, cx) >= 3:
                        break
                # 若这条链长度很短，并且终止于分叉附近，则剪掉
                if 2 <= len(path) <= max_len and neighbor_count(path[-1][0], path[-1][1]) >= 3:
                    for y, x in path[:-1]:
                        sk[y, x] = 0
                    removed_any = True
        return sk

    def _corridor_centerlines(self, skeleton: np.ndarray, corridor_area: np.ndarray) -> Tuple[List[Dict[str, any]], Optional[np.ndarray]]:
        """从 corridor skeleton 提取中心线 polyline，并估计局部宽度"""
        centerlines: List[Dict[str, any]] = []
        if skeleton is None or skeleton.size == 0 or not skeleton.any():
            return centerlines, None
        graph, nodes = self._skeleton_to_graph(skeleton)
        if graph.number_of_edges() == 0:
            return centerlines, None
        dist = cv2.distanceTransform((corridor_area > 0).astype(np.uint8), cv2.DIST_L2, 5)
        node_coords = {node["id"]: (node["x"], node["y"]) for node in nodes}
        overlay = cv2.cvtColor((corridor_area > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        for src, dst, data in graph.edges(data=True):
            path = self._generate_corridor_path(node_coords[src], node_coords[dst], skeleton, skeleton.shape)
            widths = []
            for x, y in path:
                if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1]:
                    widths.append(float(dist[y, x] * 2.0))
            width_est = float(np.median(widths)) if widths else None
            centerlines.append(
                {
                    "source_node_id": int(src),
                    "target_node_id": int(dst),
                    "polyline": [{"x": float(x), "y": float(y)} for x, y in path],
                    "width_estimate": width_est,
                }
            )
            for i in range(1, len(path)):
                cv2.line(overlay, path[i - 1], path[i], (0, 200, 0), 2, lineType=cv2.LINE_AA)
        return centerlines, overlay

    def _extract_rooms(self, free_space: np.ndarray, corridor_area: np.ndarray) -> List[Dict[str, any]]:
        """将 free_space - corridor 视为房间连通域，输出 polygon"""
        room_mask = (free_space > 0) & (corridor_area == 0)
        room_mask = remove_small_objects(room_mask, min_size=int(room_mask.size * 0.0005))
        labeled = label(room_mask)
        rooms: List[Dict[str, any]] = []
        for region in regionprops(labeled):
            minr, minc, maxr, maxc = region.bbox
            polygon = [
                {"x": int(minc), "y": int(minr)},
                {"x": int(maxc), "y": int(minr)},
                {"x": int(maxc), "y": int(maxr)},
                {"x": int(minc), "y": int(maxr)},
            ]
            rooms.append(
                {
                    "id": int(region.label),
                    "polygon": polygon,
                    "bbox": [int(minc), int(minr), int(maxc), int(maxr)],
                    "label": None,
                    "confidence": 0.0,
                    "door_points": [],
                }
            )
        return rooms

    def _bind_rooms_with_text(self, rooms: List[Dict[str, any]], ocr_results: List[OCRResult], centerlines: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """房号绑定 + 门点推断"""
        if not rooms:
            return rooms
        for ocr in ocr_results:
            room_text = self._normalize_room_text(ocr.text)
            if not room_text:
                continue
            cx, cy = ocr.center_x, ocr.center_y
            best_room = None
            best_dist = float("inf")
            for room in rooms:
                x0, y0, x1, y1 = room["bbox"]
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    best_room = room
                    break
                # 距离 bbox 最近边
                dx = max(x0 - cx, 0, cx - x1)
                dy = max(y0 - cy, 0, cy - y1)
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_room = room
            if best_room:
                best_room["label"] = room_text
                best_room["confidence"] = max(best_room.get("confidence", 0.0), float(ocr.confidence))
        # 门点：房间与 corridor 接触的中点
        max_x = max(r["bbox"][2] for r in rooms)
        max_y = max(r["bbox"][3] for r in rooms)
        for c in centerlines:
            for p in c["polyline"]:
                max_x = max(max_x, int(p["x"]))
                max_y = max(max_y, int(p["y"]))
        corridor_mask = np.zeros((max(max_y + 5, 1), max(max_x + 5, 1)), dtype=np.uint8)
        # corridor mask 取中心线附近
        for c in centerlines:
            pts = [(int(p["x"]), int(p["y"])) for p in c["polyline"]]
            for i in range(1, len(pts)):
                cv2.line(corridor_mask, pts[i - 1], pts[i], 255, 5)
        for room in rooms:
            x0, y0, x1, y1 = room["bbox"]
            mask = np.zeros_like(corridor_mask)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            boundary = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
            contact = cv2.bitwise_and(boundary, corridor_mask)
            ys, xs = np.where(contact > 0)
            if len(xs) == 0:
                continue
            pts = list(zip(xs.tolist(), ys.tolist()))
            # 采样若干门点
            step = max(1, len(pts) // 4)
            sampled = pts[::step][:4]
            room["door_points"] = [{"x": int(p[0]), "y": int(p[1])} for p in sampled]
        return rooms

    def _detect_facilities(
        self,
        image_bgr: np.ndarray,
        ocr_results: List[OCRResult],
        templates: Dict[str, np.ndarray],
        palettes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> List[Dict[str, any]]:
        """融合文本/模板/颜色位置的设施识别"""
        facilities: List[Dict[str, any]] = []
        # 文本触发
        for r in ocr_results:
            facility_type = self._identify_facility_type(r.text)
            if facility_type != FacilityType.UNKNOWN:
                facilities.append(
                    {
                        "type": facility_type.value,
                        "center": {"x": r.center_x, "y": r.center_y},
                        "text": r.text,
                        "evidence_sources": ["text"],
                        "confidence": float(r.confidence),
                    }
                )
        # 模板匹配
        for key, tmpl in templates.items():
            if tmpl is None or tmpl.size == 0:
                continue
            res = cv2.matchTemplate(image_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.6)
            used_points: List[Tuple[int, int]] = []
            for pt in zip(*loc[::-1]):
                cx = int(pt[0] + tmpl.shape[1] / 2)
                cy = int(pt[1] + tmpl.shape[0] / 2)
                if any(math.hypot(cx - px, cy - py) < 20 for px, py in used_points):
                    continue
                used_points.append((cx, cy))
                facilities.append(
                    {
                        "type": FacilityType.EXIT.value if key == "exit" else FacilityType.FIRE_HYDRANT.value,
                        "center": {"x": cx, "y": cy},
                        "text": None,
                        "evidence_sources": ["icon", "legend"],
                        "confidence": float(np.max(res)),
                    }
                )
        return facilities
