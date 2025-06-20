import pandas as pd
import numpy as np  # 关键：添加NumPy导入
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. 数据加载与基本检查
output_path = r"D:\course2024\AI\ass-AI\project2\CNN-ConvLSTM-Pm2.5-standard\data\merged_stations_pm.csv"
merged_data = pd.read_csv(output_path)

print("数据基本信息：")
print(merged_data.info())

# 2. 处理缺失值（以PM2.5为例）
numeric_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
for col in numeric_cols:
    missing_count = merged_data[col].isna().sum()
    if missing_count > 0:
        print(f"{col} 缺失值数量: {missing_count}, 采用均值填充")
        merged_data[col] = merged_data[col].fillna(merged_data[col].mean())

# 3. 处理异常值（使用NumPy的where函数）
print("处理PM2.5和PM10异常值...")
pm25_median = merged_data['PM2.5'].median()
pm10_median = merged_data['PM10'].median()

merged_data['PM2.5'] = np.where(merged_data['PM2.5'] > 500, pm25_median, merged_data['PM2.5'])
merged_data['PM10'] = np.where(merged_data['PM10'] > 500, pm10_median, merged_data['PM10'])

# 4. 特征工程（时间特征提取）
print("提取时间特征...")
merged_data['time'] = pd.to_datetime(merged_data['time'])
merged_data['year'] = merged_data['time'].dt.year
merged_data['month'] = merged_data['time'].dt.month
merged_data['day'] = merged_data['time'].dt.day
merged_data['hour'] = merged_data['time'].dt.hour

# 5. 数据标准化
print("标准化数值特征...")
scaler = StandardScaler()
merged_data[numeric_cols] = scaler.fit_transform(merged_data[numeric_cols])

# 6. 保存处理后的数据
cleaned_path = output_path.replace('.csv', '_cleaned.csv')
merged_data.to_csv(cleaned_path, index=False)
print(f"清洗后数据已保存至：{cleaned_path}")

