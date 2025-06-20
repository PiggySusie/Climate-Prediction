import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 加载模型和数据
# 加载模型和数据
model = load_model("processed_data/conv_lstm_multistation_v2.h5")
X = np.load("processed_data/X_cnn_clean_mul_new.npy")        # (N, 6, 3, 4, 5)
y_true = np.load("processed_data/y_cnn_clean_mul_new.npy")   # (N, 3, 4)

# 选择一个时间步
i = 123  # 可以换成 0~len(X)-1 任意整数

# 模型预测
y_pred = model.predict(X[i:i+1])[0]  # (3, 4)
y_real = y_true[i]                   # (3, 4)

# 可视化预测热图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(y_real, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='PM2.5')
plt.title("Ground Truth PM2.5 (3×4)")

plt.subplot(1, 2, 2)
plt.imshow(y_pred, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='PM2.5')
plt.title("Predicted PM2.5 (3×4)")

plt.suptitle(f"Time Step #{i} — PM2.5 Prediction vs Reality", fontsize=14)
plt.tight_layout()
plt.show()


# 差值图（误差热图)
plt.imshow(y_pred - y_real, cmap='bwr', interpolation='nearest')
plt.colorbar(label='Prediction Error')
plt.title("Prediction Error Heatmap")
plt.show()
