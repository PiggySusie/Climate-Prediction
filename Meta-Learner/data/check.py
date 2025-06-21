import pandas as pd

# 读取 CSV 文件
scaled_df = pd.read_csv('beijing_climate_scaled.csv')
series_df = pd.read_csv('beijing_climate_series.csv')

# 查看数据的前几行
print("Scaled Data Preview:")
print(scaled_df.head())

print("\nSeries Data Preview:")
print(series_df.head())
