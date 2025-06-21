import xarray as xr
import numpy as np
import json
import os

file_path = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\outputs\2016-08-30-17-00to2016-09-08-13-00\output_surface_2016-09-01-17-00.nc"
output_json = "data_temperature.json"

with xr.open_dataset(file_path) as ds:
    t2m = ds['temperature_2m'].values - 273.15  # 转为 °C
    lon = ds.longitude.values.tolist()
    lat = ds.latitude.values.tolist()
    t2m_list = t2m.tolist()

    json_data = {
        "lon": lon,
        "lat": lat,
        "temperature": t2m_list
    }

with open(output_json, "w") as f:
    json.dump(json_data, f)

print("数据已导出为 data_temperature.json")
