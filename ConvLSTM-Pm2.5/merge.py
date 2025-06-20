import pandas as pd
import os
from tqdm import tqdm

# 设置文件夹路径
data_dir = r"D:\course2024\AI\ass-AI\project2\CNN-ConvLSTM-Pm2.5-standard\PRSA_Data_20130301-20170228"

# 所有特征列
features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# 合并后的结果列表
all_data = []

# 遍历所有站点文件
for fname in tqdm(os.listdir(data_dir)):
    if not fname.endswith(".csv") and not fname.endswith(".xlsx"):
        continue

    fpath = os.path.join(data_dir, fname)
    df = pd.read_csv(fpath) if fname.endswith(".csv") else pd.read_excel(fpath)

    # 构造统一时间字段
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No", "year", "month", "day", "hour"])

    # 标准化字段（缺失字段填 NaN）
    for col in features:
        if col not in df.columns:
            df[col] = pd.NA

    # 保留必要字段
    df = df[["time", "station"] + features]
    all_data.append(df)

# 合并所有站点
merged = pd.concat(all_data)

# 设置统一时间索引，按时间 + 站点排序
merged = merged.sort_values(["time", "station"]).reset_index(drop=True)

# 保存合并结果
output_path = r"D:\course2024\AI\ass-AI\project2\CNN-ConvLSTM-Pm2.5-standard\merged_stations_pm.csv"
merged.to_csv(output_path, index=False)
print(f"所有站点已合并并保存：{output_path}")
