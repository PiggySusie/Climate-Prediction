import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dropout, Conv2D, Reshape, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 配置路径 ===
output_dir = r"D:\course2024\AI\ass-AI\project2\CNN-ConvLSTM-Pm2.5-standard\processed_data"
X_path = os.path.join(output_dir, "X_cnn_clean_mul_new.npy")
y_path = os.path.join(output_dir, "y_cnn_clean_mul_new.npy")
model_save_path = os.path.join(output_dir, "conv_lstm_multistation_v2.h5")

# === 加载数据 ===
try:
    X = np.load(X_path)  # shape: (N, 6, 3, 4, 5)
    y = np.load(y_path)  # shape: (N, 3, 4)
    print(f"数据加载完成: X.shape={X.shape}, y.shape={y.shape}")
except FileNotFoundError:
    print(f"错误: 找不到数据文件\n请检查路径: {X_path}\n{y_path}")
    exit()

# 检查 NaN
if np.isnan(X).any() or np.isnan(y).any():
    print("警告: 检测到 NaN 值，请检查数据处理结果！")

# 拆分训练集和测试集（保持时间顺序，不打乱）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"数据拆分完成: 训练集={X_train.shape[0]}, 测试集={X_test.shape[0]}")

# === 构建优化的模型 ===
def build_conv_lstm_model(input_shape, output_shape=(3, 4)):
    """构建优化的ConvLSTM模型，增加残差连接和多尺度特征"""
    inputs = Input(shape=input_shape)
    
    # 第一层ConvLSTM
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        activation='tanh',
        padding='same',
        return_sequences=True,
        name='conv_lstm_1'
    )(inputs)
    x = BatchNormalization()(x)
    
    # 第二层ConvLSTM
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        activation='tanh',
        padding='same',
        return_sequences=False,
        name='conv_lstm_2'
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 多尺度特征提取
    # 大尺度特征
    x1 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv2d_large'
    )(x)
    x1 = BatchNormalization()(x1)
    
    # 小尺度特征
    x2 = Conv2D(
        filters=32,
        kernel_size=(1, 1),
        activation='relu',
        padding='same',
        name='conv2d_small'
    )(x)
    x2 = BatchNormalization()(x2)
    
    # 特征融合
    x = Concatenate()([x1, x2])
    x = Dropout(0.2)(x)
    
    # 输出层
    x = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        activation='linear',
        padding='same',
        name='output_conv'
    )(x)
    
    # 调整输出形状
    outputs = Reshape(output_shape)(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs, name='MultiStation_ConvLSTM')
    return model

# 构建模型
input_shape = X.shape[1:]  # (6, 3, 4, 5)
model = build_conv_lstm_model(input_shape)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

model.summary()

# === 定义回调函数 ===
callbacks = [
    ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# === 训练模型 ===
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# === 可视化训练过程 ===
def plot_training_history(history):
    """绘制训练过程图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制Loss曲线
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制MAE曲线
    ax2.plot(history.history['mean_absolute_error'], label='Train MAE')
    ax2.plot(history.history['val_mean_absolute_error'], label='Val MAE')
    ax2.set_title('平均绝对误差曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (K)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.show()

plot_training_history(history)

# === 评估模型 ===
def evaluate_model(model, X_test, y_test):
    """评估模型并打印指标"""
    results = model.evaluate(X_test, y_test, batch_size=32)
    print(f"测试集评估结果: Loss={results[0]:.4f}, MAE={results[1]:.4f}")
    
    # 计算每个站点的MAE
    y_pred = model.predict(X_test)
    station_mae = []
    for h in range(3):
        for w in range(4):
            mae = np.mean(np.abs(y_pred[:, h, w] - y_test[:, h, w]))
            station_mae.append((f"站点[{h},{w}]", mae))
    
    # 打印各站点MAE
    station_mae.sort(key=lambda x: x[1], reverse=True)
    print("\n各站点MAE:")
    for station, mae in station_mae:
        print(f"{station}: {mae:.4f}")
    
    return results, y_pred

results, y_pred = evaluate_model(model, X_test, y_test)

# === 可视化预测结果 ===
def visualize_predictions(y_test, y_pred, station_grid, n_samples=5):
    """可视化预测结果"""
    plt.figure(figsize=(18, 6))
    
    # 随机选择n_samples个样本可视化
    indices = np.random.choice(len(y_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, n_samples, i+1)
        
        # 实际值热图
        plt.subplot(1, n_samples, i+1)
        plt.title(f'样本 {idx}')
        im = plt.imshow(y_test[idx], cmap='viridis')
        plt.colorbar(im)
        
        # 添加站点名称
        for h in range(3):
            for w in range(4):
                plt.text(w, h, station_grid[h][w], ha='center', va='bottom', color='w')
        
        # 显示预测值
        for h in range(3):
            for w in range(4):
                plt.text(w, h, f'{y_pred[idx, h, w]:.1f}', ha='center', va='top', color='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_visualization.png'), dpi=300)
    plt.show()

# 假设station_grid已定义
station_grid = np.array([
    ["Huairou", "Shunyi", "Changping", "Dingling"],
    ["Wanliu", "Gucheng", "Aotizhongxin", "Dongsi"],
    ["Guanyuan", "Wanshouxigong", "Tiantan", "Nongzhanguan"]
])
visualize_predictions(y_test, y_pred, station_grid)

print("模型训练与评估完成！")