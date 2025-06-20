import os
import json
import xarray as xr
import numpy as np

# 输入/输出路径
input_dir = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\outputs\2016-08-30-17-00to2016-09-08-13-00"
output_dir = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\web\data"
os.makedirs(output_dir, exist_ok=True)

file_list = []

for fname in os.listdir(input_dir):
    if fname.endswith(".nc") and fname.startswith("output_surface_"):
        path = os.path.join(input_dir, fname)
        try:
            with xr.open_dataset(path) as ds:
                lon = ds.longitude.values.tolist()
                lat = ds.latitude.values.tolist()
                t2m = (ds['temperature_2m'].values - 273.15).tolist()
                msl = (ds['mean_sea_level_pressure'].values / 100).tolist()
                u10 = ds['u_component_of_wind_10m'].values.tolist()
                v10 = ds['v_component_of_wind_10m'].values.tolist()
            
            output_json = os.path.join(output_dir, fname.replace(".nc", ".json"))
            with open(output_json, "w") as f:
                json.dump({
                    "lon": lon,
                    "lat": lat,
                    "temperature_2m": t2m,
                    "mean_sea_level_pressure": msl,
                    "u_component_of_wind_10m": u10,
                    "v_component_of_wind_10m": v10
                }, f)
            file_list.append(fname.replace(".nc", ".json"))
            print(f"已导出: {fname}")
        except Exception as e:
            print(f"跳过 {fname}: {e}")

# 写入下拉菜单数据列表
with open(os.path.join(output_dir, "data_list.json"), "w") as f:
    json.dump(file_list, f)

print("\n所有文件已完成转换，已保存 data_list.json")
