import os
os.environ['PROJ_LIB'] = r"C:\Users\ASUS\.conda\envs\pangu\Library\share\proj"


import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# 输入数据
json_dir = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\web\data"
shapefile = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\web\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp"
output_dir = os.path.join(json_dir, "avg_temp_by_country")
os.makedirs(output_dir, exist_ok=True)

# 加载国界
world = gpd.read_file(shapefile).to_crs("EPSG:4326")

# 遍历 JSON 文件
for fname in os.listdir(json_dir):
    if not fname.endswith(".json") or fname == "data_list.json":
        continue

    print(f"🔍 正在处理：{fname}")
    with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as f:
        data = json.load(f)

    lon = data["lon"]
    lat = data["lat"]
    t2m = np.array(data["temperature_2m"])  # shape: [lat, lon]

    lon2d, lat2d = np.meshgrid(lon, lat)
    points = [Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())]
    temps = t2m.ravel()

    gdf = gpd.GeoDataFrame({"temp": temps}, geometry=points, crs="EPSG:4326")

    result = []
    for _, row in world.iterrows():
        country_name = row["NAME"]
        shape = row["geometry"]
        mask = gdf.within(shape)
        values = gdf.loc[mask, "temp"]

        if len(values) > 0:
            avg_temp = float(np.mean(values))
            result.append({
                "name": country_name,
                "value": round(avg_temp, 2)
            })

    out_file = os.path.join(output_dir, fname)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存: {out_file}")
