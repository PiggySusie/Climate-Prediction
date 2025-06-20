import re
import json

input_path = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\web\data\world.js"
output_path = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\web\data\world.json"

# 读取 js 文件内容
with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# 用正则提取 registerMap 中的 JSON 对象（从 { 开始到最后一个 } 结束）
match = re.search(r"registerMap\(\s*'world'\s*,\s*(\{.*\})\s*\)\s*;", content, re.DOTALL)

if match:
    geojson_str = match.group(1)
    try:
        # 解析 JSON
        geojson_obj = json.loads(geojson_str)
        # 保存为 .json 文件
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(geojson_obj, out, ensure_ascii=False, indent=2)
        print(f"✅ world.json 已保存至：{output_path}")
    except json.JSONDecodeError as e:
        print("❌ JSON 解析失败，请检查 world.js 格式是否标准：", e)
else:
    print("❌ 未找到 echarts.registerMap('world', {...}); 中的 GeoJSON 数据。")
