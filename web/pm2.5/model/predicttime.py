import os
import sys
import pandas as pd
import numpy as np

from joblib import load
from tensorflow.keras.models import load_model


# === 路径设置 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.dirname(__file__)

input_csv = os.path.join(DATA_DIR, "small_sample_cleaned_201603.csv")
true_csv = os.path.join(DATA_DIR, "small_sample_201603.csv")

scaler_path = os.path.join(MODEL_DIR, "scaler_with_features.save")
model_path = os.path.join(MODEL_DIR, "conv_lstm_multistation_v2.h5")

def predict_by_time(target_time_str):
    # === 加载标准化后的输入数据 ===
    df = pd.read_csv(input_csv, parse_dates=["time"])
    df = df.sort_values("time")
    scaler, features = load(scaler_path)

    # === 准备时间序列 & 网格 ===
    time_list = df["time"].drop_duplicates().sort_values().reset_index(drop=True)
    n_steps = 6
    station_grid = np.array([
        ["Huairou", "Shunyi", "Changping", "Dingling"],
        ["Wanliu", "Gucheng", "Aotizhongxin", "Dongsi"],
        ["Guanyuan", "Wanshouxigong", "Tiantan", "Nongzhanguan"]
    ])
    H, W = station_grid.shape

    # === 目标时间检查 ===
    target_time = pd.to_datetime(target_time_str)
    if target_time not in set(time_list):
        print(f"时间 {target_time_str} 不存在")
        return
    idx = time_list[time_list == target_time].index[0]
    if idx < n_steps:
        print("时间太早，前面不足滑窗步长")
        return

    # === 构建模型输入样本 ===
    sample = np.zeros((1, n_steps, H, W, len(features)))
    for step in range(n_steps):
        t = time_list[idx - n_steps + step]
        frame = df[df["time"] == t]
        for h in range(H):
            for w in range(W):
                station = station_grid[h][w]
                row = frame[frame["station"] == station]
                sample[0, step, h, w, :] = row[features].values[0]

    # === 预测并反标准化 ===
    model = load_model(model_path)
    pred_std = model.predict(sample)[0]

    pm25_index = features.index("PM2.5")
    mean = scaler.mean_[pm25_index]
    std = scaler.scale_[pm25_index]
    pred_real = pred_std * std + mean

    print(f"\n 反标准化参数：mean={mean:.2f}, std={std:.2f}")
    print(f"\n预测 PM2.5（{target_time_str}）:")
    print(np.round(pred_real, 1))

    # === 真实值来自原始数据 === 
    df_true = pd.read_csv(true_csv, parse_dates=["time"])
    true_frame = df_true[df_true["time"] == target_time]

    true_grid = np.zeros((H, W))
    for h in range(H):
        for w in range(W):
            station = station_grid[h][w]
            row = true_frame[true_frame["station"] == station]
            true_grid[h, w] = row["PM2.5"].values[0]  # 原始单位，无需反标准化

    print(f"\n 实际 PM2.5 网格（μg/m³）:")
    print(np.round(true_grid, 1))

    # === 误差网格 ===
    error_grid = pred_real - true_grid
    print(f"\n误差网格 (预测 - 实际):")
    print(np.round(error_grid, 1))

    return pred_real, true_grid, error_grid 


#predict_by_time("2016-09-01 17:00")
#predict_by_time("2016-03-01 17:00")