#!/usr/bin/env python3
"""直接测试地图处理功能（不通过API）"""

import sys
import os
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.map_processor import MapDigitizer


def test_direct_processing(image_path: str):
    """直接测试地图处理功能"""
    
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片文件不存在: {image_path}")
        return
    
    print(f"[INFO] 正在处理图片: {image_path}")
    print("[INFO] 初始化地图处理器...")
    
    try:
        # 初始化处理器
        digitizer = MapDigitizer(use_ocr=True)
        
        # 处理消防图
        print("[INFO] 开始处理...")
        result = digitizer.process_fire_map(
            image_path=image_path,
            apply_perspective=True,
            extract_ocr=True
        )
        
        # 打印结果统计
        print("\n" + "="*60)
        print("[OK] 处理成功！")
        print("="*60)
        print(f"节点数量: {len(result['nodes'])}")
        print(f"边数量: {len(result['edges'])}")
        print(f"OCR结果数量: {len(result.get('ocr_results', []))}")
        
        # 统计节点类型
        node_types = {}
        for node in result['nodes']:
            node_type = node.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("\n节点类型统计:")
        for node_type, count in sorted(node_types.items()):
            print(f"  {node_type}: {count}")
        
        # 统计设施类型
        facilities = {}
        for node in result['nodes']:
            if node.get('facility_type'):
                facility_type = node['facility_type']
                facilities[facility_type] = facilities.get(facility_type, 0) + 1
        
        if facilities:
            print("\n重要设施统计:")
            for facility_type, count in sorted(facilities.items()):
                print(f"  {facility_type}: {count}")
        
        # 显示OCR结果
        if result.get('ocr_results'):
            print("\nOCR识别结果（前10个）:")
            for i, ocr in enumerate(result['ocr_results'][:10], 1):
                print(f"  {i}. {ocr['text']} (置信度: {ocr['confidence']:.2f}, "
                      f"位置: ({ocr['center_x']}, {ocr['center_y']}))")
        
        # 显示重要节点示例
        important_nodes = [
            n for n in result['nodes'] 
            if n.get('node_type') in ['intersection', 'corner', 'facility']
        ]
        
        if important_nodes:
            print(f"\n重要节点示例（显示前10个）:")
            for node in important_nodes[:10]:
                info = f"  节点{node['id']}: {node.get('node_type', 'unknown')}"
                if node.get('facility_type'):
                    info += f" ({node['facility_type']})"
                if node.get('ocr_text'):
                    info += f" - OCR: {node['ocr_text']}"
                if node.get('angle'):
                    info += f" - 角度: {node['angle']:.1f}°"
                info += f" - 度数: {node.get('degree', 'N/A')}"
                info += f" - 置信度: {node.get('confidence', 0):.2f}"
                info += f" - 位置: ({node['x']}, {node['y']})"
                print(info)
        
        # 保存结果到JSON文件
        output_file = os.path.splitext(image_path)[0] + "_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_map_processor_direct.py <图片路径>")
        print("示例: python test_map_processor_direct.py ../demo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_direct_processing(image_path)
