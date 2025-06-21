import pandas as pd

# 读取 CSV 文件
scaled_df = pd.read_csv('beijing_climate_series.csv')

# 将 'time' 列转换为日期时间格式
scaled_df['time'] = pd.to_datetime(scaled_df['time'])

# 筛选 2016 年 3 月 1 日到 3 月 10 日的数据
filtered_scaled_df = scaled_df[(scaled_df['time'] >= '2016-03-01') & (scaled_df['time'] <= '2016-03-10')]

# 查看筛选后的数据
print(filtered_scaled_df.head())

# 保存筛选后的数据
filtered_scaled_df.to_csv('filtered_series_data_20160301_to_20160310.csv', index=False)
