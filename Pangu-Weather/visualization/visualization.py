import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'

# 构建文件路径
base_dir = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\outputs\2025-05-31-13-00to2025-06-10-10-00"
file_name = "output_surface_2025-06-03-13-00.nc"
file_path = os.path.join(base_dir, file_name)

# 打开NetCDF文件
ds = xr.open_dataset(file_path)
print("文件中包含的变量:", list(ds.variables.keys()))

# 处理平均海平面气压（转换为hPa）
msl_data = ds['mean_sea_level_pressure'].values
msl_hpa = msl_data / 100  # Pa转换为hPa
print("平均海平面气压数据范围:", msl_hpa.min(), msl_hpa.max())

# 处理2米温度（正确获取单位）
t2m_var = ds['temperature_2m']
t2m_data = t2m_var.values
units = t2m_var.attrs.get('units', 'K')  # 获取单位，默认为K

# 转换为摄氏度
if units == 'K':
    t2m_celsius = t2m_data - 273.15
    print("2米温度数据范围 (°C):", t2m_celsius.min(), t2m_celsius.max())
else:
    t2m_celsius = t2m_data
    print(f"2米温度数据范围 ({units}):", t2m_celsius.min(), t2m_celsius.max())

# 可视化平均海平面气压（hPa）
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 添加地理特征
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# 设置色阶（根据hPa数据范围）
levels = np.linspace(msl_hpa.min(), msl_hpa.max(), 20)
contour = ax.contourf(
    ds.longitude, ds.latitude, msl_hpa,
    levels=levels, cmap='jet', transform=ccrs.PlateCarree()
)

# 添加色标和标题
cbar = plt.colorbar(contour, ax=ax, shrink=0.7, pad=0.05)
cbar.set_label('平均海平面气压 (hPa)', fontsize=12)
ax.set_title(f"平均海平面气压分布 - 2025-06-03 13:00", fontsize=14)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.tight_layout()
plt.show()

# 可视化2米温度（°C）
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# 设置温度色阶
temp_levels = np.linspace(t2m_celsius.min(), t2m_celsius.max(), 15)
im = ax.contourf(
    ds.longitude, ds.latitude, t2m_celsius,
    levels=temp_levels, cmap='rainbow', transform=ccrs.PlateCarree()
)

# 添加色标和标题
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
cbar.set_label('温度 (°C)', fontsize=12)
ax.set_title(f"2米温度分布 - 2025-06-03 13:00", fontsize=14)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.tight_layout()
plt.show()