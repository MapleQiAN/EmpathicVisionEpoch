from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from paddleocr import PaddleOCR
from skimage.morphology import skeletonize


class OCRResult:
    """OCR识别结果"""
    def __init__(self, text: str, bbox: List[List[int]], confidence: float):
        self.text = text
        self.bbox = bbox  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        self.confidence = confidence
        self.center_x = int(np.mean([p[0] for p in bbox]))
        self.center_y = int(np.mean([p[1] for p in bbox]))


class MapDigitizer:
    """Digitize a fire-escape map image into a topological graph."""

    def __init__(
        self,
        adaptive_block_size: int = 35,
        adaptive_c: int = 10,
        morph_kernel_size: int = 3,
        use_ocr: bool = True,
        ocr_lang: str = "ch",  # ch: 中文, en: 英文
    ) -> None:
        # Tuning knobs for thresholding and denoising.
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.use_ocr = use_ocr
        self.ocr = None
        if use_ocr:
            try:
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

        result = {
            "nodes": nodes,
            "edges": edges,
            "ocr_results": ocr_dicts,
        }
        
        if perspective_matrix is not None:
            result["perspective_matrix"] = perspective_matrix.tolist()

        return result

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
