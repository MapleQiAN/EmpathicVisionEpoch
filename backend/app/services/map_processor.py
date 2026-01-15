from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


class MapDigitizer:
    """Digitize a fire-escape map image into a topological graph."""

    def __init__(
        self,
        adaptive_block_size: int = 35,
        adaptive_c: int = 10,
        morph_kernel_size: int = 3,
    ) -> None:
        # Tuning knobs for thresholding and denoising.
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size

    def process_fire_map(self, image_path: str) -> Dict[str, List[Dict[str, int | float]]]:
        """Process a fire-escape map image and return nodes + edges.

        The output dictionary contains:
          - nodes: list of {id, x, y}
          - edges: list of {source_node_id, target_node_id, length, weight}
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Step 1: Preprocessing - grayscale and adaptive threshold for line isolation.
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

        # Step 2: Skeletonization - reduce line thickness to 1px for graph tracing.
        skeleton = skeletonize(binary > 0)

        # Step 3: Graph extraction from skeleton pixels.
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

        return {"nodes": nodes, "edges": edges}

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
