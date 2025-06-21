import cdsapi
import numpy as np
import netCDF4 as nc
import os
from datetime import datetime

c = cdsapi.Client()
date_time = datetime(
    year=2025, 
    month=5, 
    day=31,
    hour=13,
    minute=0)

# 获取项目根目录（假设脚本位于项目根目录下）
project_root = os.path.dirname(os.path.abspath(__file__))

forecast_dir = os.path.join(
    project_root,
    "forecasts", 
    date_time.strftime("%Y-%m-%d-%H-%M"),
)

# 确认目录是否存在
if not os.path.exists(forecast_dir):
    print(f"错误：目录 {forecast_dir} 不存在！")
else:
    # 转换地面数据为npy
    surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
    try:
        with nc.Dataset(os.path.join(forecast_dir, 'surface.nc')) as nc_file:
            surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
            surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
            surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
            surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
        np.save(os.path.join(forecast_dir, 'input_surface.npy'), surface_data)
        print("地面数据已成功转换为npy格式")
    except FileNotFoundError:
        print(f"错误：在 {forecast_dir} 目录下找不到 surface.nc 文件")
    except Exception as e:
        print(f"处理 surface.nc 时出错: {e}")

    # 转换高空数据为npy
    upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
    try:
        with nc.Dataset(os.path.join(forecast_dir, 'upper.nc')) as nc_file:
            upper_data[0] = (nc_file.variables['z'][:]).astype(np.float32)
            upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
            upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
            upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
            upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
        np.save(os.path.join(forecast_dir, 'input_upper.npy'), upper_data)
        print("高空数据已成功转换为npy格式")
    except FileNotFoundError:
        print(f"错误：在 {forecast_dir} 目录下找不到 upper.nc 文件")
    except Exception as e:
        print(f"处理 upper.nc 时出错: {e}")