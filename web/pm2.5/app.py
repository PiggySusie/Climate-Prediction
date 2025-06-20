from flask import Flask, request, jsonify, send_from_directory
import sys
sys.path.append("model")  # 把 model 文件夹加入模块搜索路径
from predicttime import predict_by_time
import traceback
import os
import pandas as pd

app = Flask(__name__)

# === 路径设置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹
DATA_DIR = os.path.join(BASE_DIR, "data")
csv_path = os.path.join(DATA_DIR, "merged_stations_pm_cleaned.csv")


# ==== 主页导航 ====
@app.route("/")
def home():
    return send_from_directory("..", "index.html")  # 返回上一级的导航页

# ==== 静态资源映射 ====
@app.route("/pm2.5/<path:filename>")
def serve_pm25(filename):
    return send_from_directory(".", filename)

@app.route("/pangu/<path:filename>")
def serve_pangu(filename):
    return send_from_directory("../pangu", filename)


@app.route("/valid_times")
def valid_times():
    time_list = pd.read_csv(csv_path, parse_dates=["time"])["time"]
    return jsonify([t.strftime("%Y-%m-%d %H:%M") for t in sorted(time_list)])


# 预测接口
@app.route("/predict")
def predict():
    time_str = request.args.get("time")
    print(f"请求预测时间: {time_str}")
    try:
        # 获取预测网格
        pred_real, _, _ = predict_by_time(time_str)  # 获取预测值网格
        return jsonify({"time": time_str, "grid": pred_real.tolist()})  # 将预测值转为列表
    except Exception as e:
        print("后端预测时出错:")
        traceback.print_exc()  # 打印详细错误堆栈
        return jsonify({"error": str(e)}), 500

# 加载前端页面（index.html）和静态资源
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
