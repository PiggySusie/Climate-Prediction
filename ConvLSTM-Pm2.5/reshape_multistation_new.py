import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
import json
from tqdm import tqdm

# === 配置 ===
csv_path = r"D:\course2024\AI\ass-AI\project2\CNN-ConvLSTM-Pm2.5-standard\data\merged_stations_pm.csv"
features = ["PM2.5", "PM10", "TEMP", "PRES", "WSPM"]
n_steps = 6
station_grid = np.array([
    ["Huairou", "Shunyi", "Changping", "Dingling"],
    ["Wanliu", "Gucheng", "Aotizhongxin", "Dongsi"],
    ["Guanyuan", "Wanshouxigong", "Tiantan", "Nongzhanguan"]
])
H, W = station_grid.shape
output_dir = r"D:\course2024\AI\ass-AI\project2\CNN-ConvLSTM-Pm2.5-standard\processed_data"

# === Step 1: 读取 + 标准化 ===
print("读取并标准化数据...")
df = pd.read_csv(csv_path, parse_dates=["time"])
df = df.sort_values("time")

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 保存标准化器
dump((scaler, features), f"{output_dir}/scaler_with_features.save")
print("scaler已保存，包含特征信息")

# === Step 2: 准备时间序列 ===
time_list = df["time"].drop_duplicates().sort_values().reset_index(drop=True)
print(f"时间序列长度: {len(time_list)}")

# === Step 3: 构建站点索引映射 ===
print("构建站点索引映射...")
station_to_idx = {(h, w): station_grid[h][w] for h in range(H) for w in range(W)}
idx_to_station = {v: k for k, v in station_to_idx.items()}

# === Step 4: 生成样本 ===
print("开始生成样本...")
X, y = [], []

for idx in tqdm(range(n_steps, len(time_list))):
    sample = np.zeros((n_steps, H, W, len(features)))
    target_grid = np.zeros((H, W))
    valid = True

    # 处理输入样本
    for step in range(n_steps):
        t = time_list[idx - n_steps + step]
        time_frame = df[df["time"] == t]
        if time_frame.empty:
            valid = False
            break
        
        for (h, w), station in station_to_idx.items():
            row = time_frame[time_frame["station"] == station]
            if row.empty:
                valid = False
                break
            sample[step, h, w, :] = row[features].values[0]
        if not valid:
            break
    
    if not valid:
        continue

    # 处理目标值
    target_time = time_list[idx]
    target_frame = df[df["time"] == target_time]
    if target_frame.empty:
        continue
    
    for (h, w), station in station_to_idx.items():
        row = target_frame[target_frame["station"] == station]
        if row.empty:
            valid = False
            break
        target_grid[h, w] = row["PM2.5"].values[0]
    
    if valid:
        X.append(sample)
        y.append(target_grid)

X = np.array(X)
y = np.array(y)
print(f"初始构建完成: X.shape = {X.shape}, y.shape = {y.shape}")

# === Step 5: 清洗 NaN 样本 ===
print("清洗缺失值样本...")
x_nan = np.isnan(X).any(axis=(1, 2, 3, 4)).sum()
y_nan = np.isnan(y).any(axis=(1, 2)).sum()
both_nan = (np.isnan(X).any(axis=(1, 2, 3, 4)) & np.isnan(y).any(axis=(1, 2))).sum()

print(f"X中包含NaN的样本数: {x_nan}")
print(f"y中包含NaN的样本数: {y_nan}")
print(f"X和y都包含NaN的样本数: {both_nan}")
print(f"理论有效样本数: {len(X) - x_nan - y_nan + both_nan}")

valid_indices = ~np.isnan(X).any(axis=(1, 2, 3, 4)) & ~np.isnan(y).any(axis=(1, 2))
X_clean = X[valid_indices]
y_clean = y[valid_indices]
print(f" 清洗后: X.shape = {X_clean.shape}, y.shape = {y_clean.shape}")

# === Step 6: 保存样本及元数据 ===
print("保存处理后的样本...")
metadata = {
    "features": features,
    "n_steps": n_steps,
    "station_grid": station_grid.tolist(),
    "X_shape": X_clean.shape,
    "y_shape": y_clean.shape,
    "original_sample_count": len(X),
    "clean_sample_count": len(X_clean),
    "nan_removed_count": len(X) - len(X_clean)
}

# 保存元数据
with open(f"{output_dir}/sample_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# 保存样本数据
np.save(f"{output_dir}/X_cnn_clean_mul_new.npy", X_clean)
np.save(f"{output_dir}/y_cnn_clean_mul_new.npy", y_clean)
print("清洗数据及元数据已保存")