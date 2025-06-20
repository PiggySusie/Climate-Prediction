import numpy as np
import matplotlib.pyplot as plt

# 加载数据
X = np.load("X_cnn_clean.npy")
y = np.load("y_cnn_clean.npy")

# 加载模型
from tensorflow.keras.models import load_model
model = load_model("final_convlstm_model.h5")

# 预测
y_pred = model.predict(X).ravel()

# 可视化
plt.figure(figsize=(12, 5))
plt.plot(y[:200], label="Actual PM2.5", linewidth=2)
plt.plot(y_pred[:200], label="Predicted PM2.5", linewidth=2)
plt.title("ConvLSTM2D Forecast vs Ground Truth (First 200 Hours)")
plt.xlabel("Time Step (hour)")
plt.ylabel("PM2.5 (Standardized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
