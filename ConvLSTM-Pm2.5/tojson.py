import os
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
import json

# === 新路径设置 ===
output_dir = r"D:\course2024\AI\ass-AI\project2\CNN-ConvLSTM-Pm2.5-standard\processed_data"
X_path = os.path.join(output_dir, "X_cnn_clean_mul_new.npy")
y_path = os.path.join(output_dir, "y_cnn_clean_mul_new.npy")
model_save_path = os.path.join(output_dir, "conv_lstm_multistation_v2.h5")
scaler_path = os.path.join(output_dir, "scaler_with_features.save")

# === 加载模型和数据 ===
model = load_model(model_save_path)
X = np.load(X_path)
y_true = np.load(y_path)

# === 加载 scaler 和列名
scaler, feature_names = load(scaler_path)
pm25_index = feature_names.index("PM2.5")  # 从保存的列名中找 index
mean = scaler.mean_[pm25_index]
std = scaler.scale_[pm25_index]

frames = []

for i in range(len(X)):
    y_pred = model.predict(X[i:i+1])[0]
    y_real = y_true[i]

    pred_real = y_pred * std + mean
    true_real = y_real * std + mean

    for r in range(3):
        for c in range(4):
            frames.append({
                "time_index": i,
                "row": r,
                "col": c,
                "predicted_pm25": float(pred_real[r][c]),
                "true_pm25": float(true_real[r][c]),
                "error": float(pred_real[r][c] - true_real[r][c])
            })

# === 保存输出 ===
os.makedirs("processed_data", exist_ok=True)
with open("processed_data/pm25_prediction_vs_truth_mult.json", "w", encoding="utf-8") as f:
    json.dump(frames, f, indent=2, ensure_ascii=False)


# pd.DataFrame(frames).to_csv("data/pm25_prediction_vs_truth_mult.csv", index=False)

print("已使用新模型生成预测结果并保存 JSON/CSV 到 processed_data/")
