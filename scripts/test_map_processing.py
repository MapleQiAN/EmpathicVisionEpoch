#!/usr/bin/env python3
"""测试地图处理功能的脚本"""

import sys
import os
import requests
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_map_processing(image_path: str, base_url: str = "http://localhost:8000"):
    """测试地图处理API"""
    
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片文件不存在: {image_path}")
        return
    
    url = f"{base_url}/map/process"
    
    print(f"[INFO] 正在处理图片: {image_path}")
    print(f"[INFO] API地址: {url}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {"file": f}
            params = {
                "apply_perspective": True,
                "extract_ocr": True,
                "auto_create_nodes": True,
                "building_id": "test_building",
                "floor_id": "test_floor"
            }
            
            response = requests.post(url, files=files, params=params, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("\n" + "="*50)
                print("✅ 处理成功！")
                print("="*50)
                print(f"节点数: {result['nodes_count']}")
                print(f"边数: {result['edges_count']}")
                print(f"OCR文本数: {result['ocr_count']}")
                
                # 显示OCR结果
                if result['data'].get('ocr_results'):
                    print("\nOCR识别结果:")
                    for i, ocr in enumerate(result['data']['ocr_results'][:10], 1):
                        print(f"  {i}. {ocr['text']} (置信度: {ocr['confidence']:.2f})")
                
                # 统计节点类型
                if result['data'].get('nodes'):
                    node_types = {}
                    facilities = {}
                    for node in result['data']['nodes']:
                        node_type = node.get('node_type', 'unknown')
                        node_types[node_type] = node_types.get(node_type, 0) + 1
                        
                        if node.get('facility_type'):
                            facility_type = node['facility_type']
                            facilities[facility_type] = facilities.get(facility_type, 0) + 1
                    
                    print("\n节点类型统计:")
                    for node_type, count in sorted(node_types.items()):
                        print(f"  {node_type}: {count}")
                    
                    if facilities:
                        print("\n重要设施统计:")
                        for facility_type, count in sorted(facilities.items()):
                            print(f"  {facility_type}: {count}")
                    
                    # 显示重要节点示例
                    print("\n重要节点示例（路口、转角、设施）:")
                    important_nodes = [
                        n for n in result['data']['nodes'] 
                        if n.get('node_type') in ['intersection', 'corner', 'facility']
                    ][:10]
                    for node in important_nodes:
                        info = f"  节点{node['id']}: {node.get('node_type', 'unknown')}"
                        if node.get('facility_type'):
                            info += f" ({node['facility_type']})"
                        if node.get('ocr_text'):
                            info += f" - OCR: {node['ocr_text']}"
                        if node.get('angle'):
                            info += f" - 角度: {node['angle']:.1f}°"
                        info += f" - 位置: ({node['x']}, {node['y']})"
                        print(info)
                
                # 保存结果到JSON文件
                output_file = os.path.splitext(image_path)[0] + "_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存到: {output_file}")
                
            else:
                print(f"[ERROR] 请求失败: {response.status_code}")
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print("[ERROR] 无法连接到服务器，请确保后端服务已启动:")
        print("  cd backend && uvicorn app.main:app --reload")
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_map_processing.py <图片路径> [API地址]")
        print("示例: python test_map_processing.py ../demo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    test_map_processing(image_path, base_url)
