from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Optional
import os
import tempfile
import shutil

from app.services.map_processor import MapDigitizer
from app.services.graph_store import graph_store
from app.models.graph import NodeCreate, EdgeCreate, NodeType, EdgeType

router = APIRouter(prefix="/map", tags=["map"])


@router.post("/process")
async def process_fire_map(
    file: UploadFile = File(...),
    apply_perspective: bool = True,
    extract_ocr: bool = True,
    force_ocr: bool = True,
    building_id: str = "default",
    floor_id: str = "default",
    auto_create_nodes: bool = False,
):
    """上传消防图并处理，返回拓扑图结构。
    
    参数:
    - file: 消防图图片文件
    - apply_perspective: 是否应用透视变换
    - extract_ocr: 是否提取OCR文本
    - force_ocr: 是否强制 OCR（失败直接报错，不静默禁用）
    - building_id: 建筑物ID
    - floor_id: 楼层ID
    - auto_create_nodes: 是否自动创建图节点（将OCR结果关联到节点）
    """
    # 验证文件类型
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")
    
    # 保存临时文件
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # 处理地图
        digitizer = MapDigitizer(use_ocr=extract_ocr, force_ocr=force_ocr)
        result = digitizer.process_fire_map(
            temp_file,
            apply_perspective=apply_perspective,
            extract_ocr=extract_ocr
        )
        
        # 如果启用自动创建节点，将OCR结果关联到最近的节点
        if auto_create_nodes and result.get("ocr_results"):
            _link_ocr_to_nodes(result, building_id, floor_id)
        
        return {
            "success": True,
            "nodes_count": len(result["nodes"]),
            "edges_count": len(result["edges"]),
            "ocr_count": len(result.get("ocr_results", [])),
            "data": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def _link_ocr_to_nodes(result: dict, building_id: str, floor_id: str):
    """将OCR结果关联到最近的图节点。
    
    策略：
    1. 对于每个OCR结果，找到最近的节点
    2. 如果距离小于阈值，创建锚点（Anchor）关联
    3. 根据文本内容推断节点类型（如"402"->房间，"楼梯"->楼梯）
    """
    nodes = result["nodes"]
    ocr_results = result.get("ocr_results", [])
    
    if not nodes or not ocr_results:
        return
    
    # 距离阈值（像素）
    DISTANCE_THRESHOLD = 100
    
    for ocr in ocr_results:
        ocr_x, ocr_y = ocr["center_x"], ocr["center_y"]
        text = ocr["text"].strip()
        
        # 找到最近的节点
        min_dist = float('inf')
        nearest_node_idx = None
        
        for idx, node in enumerate(nodes):
            dx = node["x"] - ocr_x
            dy = node["y"] - ocr_y
            dist = (dx * dx + dy * dy) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest_node_idx = idx
        
        # 如果距离足够近，创建节点和锚点
        if nearest_node_idx is not None and min_dist < DISTANCE_THRESHOLD:
            node = nodes[nearest_node_idx]
            
            # 推断节点类型
            node_type = _infer_node_type(text)
            
            # 创建节点（如果还没有在graph_store中）
            try:
                # 尝试通过坐标查找现有节点
                existing_nodes = graph_store.list_nodes()
                node_id = None
                
                for n in existing_nodes:
                    if (n.coord2d and 
                        abs(n.coord2d.x - node["x"]) < 5 and 
                        abs(n.coord2d.y - node["y"]) < 5):
                        node_id = n.id
                        break
                
                # 如果不存在，创建新节点
                if node_id is None:
                    from app.models.graph import Coord2D
                    node_payload = NodeCreate(
                        building_id=building_id,
                        floor_id=floor_id,
                        type=node_type,
                        label=text,
                        coord2d=Coord2D(x=float(node["x"]), y=float(node["y"])),
                        tags=[text] if text else [],
                        confidence=ocr.get("confidence", 0.5)
                    )
                    created_node = graph_store.add_node(node_payload)
                    node_id = created_node.id
                
                # 创建锚点关联OCR结果
                from app.models.graph import AnchorCreate
                anchor_payload = AnchorCreate(
                    node_id=node_id,
                    image_id="",  # TODO: 保存图片后返回image_id
                    ocr_text=[text],
                    quality={"confidence": ocr.get("confidence", 0.5)},
                    capture_meta={"bbox": ocr.get("bbox", [])}
                )
                graph_store.add_anchor(anchor_payload)
                
            except Exception as e:
                print(f"[WARN] 创建节点/锚点失败: {e}")


def _infer_node_type(text: str) -> NodeType:
    """根据OCR文本推断节点类型"""
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in ["楼梯", "stair", "楼梯间"]):
        return NodeType.stairs
    elif any(keyword in text_lower for keyword in ["电梯", "elevator", "lift"]):
        return NodeType.elevator
    elif any(keyword in text_lower for keyword in ["门", "door", "入口", "entrance"]):
        return NodeType.door
    elif any(keyword in text_lower for keyword in ["入口", "entrance", "大门"]):
        return NodeType.entrance
    elif any(keyword in text_lower for keyword in ["厕所", "toilet", "洗手间", "wc"]):
        return NodeType.landmark
    elif text.strip().isdigit() or any(char.isdigit() for char in text):
        # 房间号（如402, 301等）
        return NodeType.landmark
    else:
        return NodeType.junction
