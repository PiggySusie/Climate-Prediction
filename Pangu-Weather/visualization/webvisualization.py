import xarray as xr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import os


# 读取数据
base_dir = r"D:\course2024\AI\ass-AI\project2\Pangu-Weather-ReadyToGo\outputs\2016-08-30-17-00to2016-09-08-13-00"
# file_name = "output_surface_2025-06-03-13-00.nc"
# file_path = os.path.join(base_dir, file_name)
# ds = xr.open_dataset(file_path)


# 创建交互式应用
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # 用于生产部署

# 获取所有 nc 文件
file_options = sorted([
    f for f in os.listdir(base_dir)
    if f.endswith(".nc") and f.startswith("output_surface_")
])



# 默认值
default_file = file_options[0]

# 应用布局
app.layout = html.Div([
    html.H1("气象数据交互式可视化", style={'textAlign': 'center'}),
    html.Div([
            html.Label("选择数据文件："),
            dcc.Dropdown(
                id='file-selector',
                options=[
                    {'label': f, 'value': f}
                    for f in sorted(os.listdir(base_dir))
                    if f.endswith('.nc') and f.startswith('output_surface_')
                ],
                value=default_file,  # 默认选中的文件
                clearable=False,
                style={'width': '50%'}
            )
        ], style={'padding': '20px'}),
        
    # 选项卡组件
    dcc.Tabs(id='tabs', value='tab-msl', children=[
        # 海平面气压选项卡
        dcc.Tab(label='平均海平面气压', value='tab-msl', children=[
            html.Div([
                html.H3("平均海平面气压分布 (hPa)"),
                dcc.Graph(
                    id='msl-plot',
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["zoom2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                        "doubleClick": "reset"
                    },
                    style={"opacity": 1}
                ),
                #html.P("数据范围: {:.2f} - {:.2f} hPa".format(msl_data.min(), msl_data.max())),
                html.Div(
                dcc.Slider(
                    id='msl-contour-slider',
                    min=10,
                    max=50,
                    step=1,
                    value=20,
                    marks={i: str(i) for i in range(10, 51, 10)},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode='drag'
                ),
                style={'display': 'none'}  # 隐藏父元素
            ),
                    
            ])
        ]),
        
        # 温度选项卡
        dcc.Tab(label='2米温度', value='tab-temp', children=[
            html.Div([
                html.H3("2米温度分布 (°C)"),
                dcc.Graph(
                    id='temp-plot',
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["zoom2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                        "doubleClick": "reset"
                    },
                    style={"opacity": 1}
                ),
                #html.P("数据范围: {:.2f} - {:.2f} °C".format(t2m_data.min(), t2m_data.max())),
                dcc.RadioItems(
                    id='temp-colorscale',
                    options=[
                        {'label': '彩虹', 'value': 'Rainbow'},
                        {'label': '热图', 'value': 'Hot'},
                        {'label': '冷暖对比', 'value': 'RdBu_r'}
                    ],
                    value='Rainbow',
                    inline=True
                )
            ])
        ]),
        
        
        # 风场选项卡
        dcc.Tab(label='10米风场', value='tab-wind', children=[
            html.Div([
                html.H3("10米风场分布"),
                dcc.Graph(
                    id='wind-plot',
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["zoom2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                        "doubleClick": "reset"
                    },
                    style={"opacity": 1}
                ),
                html.P("风向和风速"),
                dcc.Slider(
                    id='wind-downsample',
                    min=5,
                    max=50,
                    step=5,
                    value=10,
                    marks={i: str(i) for i in range(5, 51, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                ),
                html.P("风箭头密度（值越大密度越低）")
            ])
        ])
    ])
])

# 回调函数 - 海平面气压图
# @app.callback(
#     Output('msl-plot', 'figure'),
#     Input('file-selector', 'value')
# )

# #     return fig
# def update_msl_plot(file_name):
#     try:
#         # 使用with语句自动管理文件关闭
#         with xr.open_dataset(os.path.join(base_dir, file_name)) as ds:
#             # 平均海平面气压 (hPa)
#             msl_data = ds['mean_sea_level_pressure'].values / 100
#             lon = ds.longitude
#             lat = ds.latitude

#             fig = go.Figure()

#             # 添加气压等高线图
#             contour = go.Contour(
#                 x=lon,
#                 y=lat,
#                 z=msl_data,
#                 contours=dict(
#                     coloring='heatmap',
#                     showlabels=False,
#                     showlines=False
#                 ),
#                 colorscale='jet',
#                 colorbar=dict(title='hPa'),
#                 hovertemplate='气压: %{z:.2f} hPa<br>经度: %{x:.2f}°<br>纬度: %{y:.2f}°<extra></extra>'
#             )
#             fig.add_trace(contour)

#             timestamp = file_name.replace("output_surface_", "").replace(".nc", "")
#             timestamp = timestamp.replace("-", " ", 3).replace("-", ":")

#             fig.update_layout(
#                 title=f"平均海平面气压分布 - {timestamp}",
#                 xaxis=dict(
#                     title="经度 (°)",
#                     showgrid=True,
#                     dtick=30,
#                     ticks="outside",
#                     tick0=0
#                 ),
#                 yaxis=dict(
#                     title="纬度 (°)",
#                     showgrid=True,
#                     dtick=30,
#                     ticks="outside",
#                     tick0=-90
#                 ),
#                 height=600,
#                 margin=dict(l=0, r=0, t=50, b=0)
#             )

#             return fig
#     except Exception as e:
#         print(f"Error in callback: {e}")
#         return go.Figure()  # 返回一个空图形避免崩溃
@app.callback(
    Output('msl-plot', 'figure'),
    Input('file-selector', 'value')
)
def update_msl_plot(file_name):
    try:
        file_path = os.path.join(base_dir, file_name)
        with xr.open_dataset(file_path) as ds:
            msl_data = ds['mean_sea_level_pressure'].values / 100
            lon = ds.longitude.values
            lat = ds.latitude.values

        fig = go.Figure()

        fig.add_trace(go.Contour(
            x=lon,
            y=lat,
            z=msl_data,
            contours=dict(coloring='heatmap', showlabels=False, showlines=False),
            colorscale='jet',
            colorbar=dict(title='hPa'),
            hovertemplate='气压: %{z:.2f} hPa<br>经度: %{x:.2f}°<br>纬度: %{y:.2f}°<extra></extra>'
        ))

        timestamp = file_name.replace("output_surface_", "").replace(".nc", "").replace("-", " ", 3).replace("-", ":")

        fig.update_layout(
            title=f"平均海平面气压分布 - {timestamp}",
            xaxis=dict(title="经度 (°)", showgrid=True, dtick=30),
            yaxis=dict(title="纬度 (°)", showgrid=True, dtick=30),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig
    except Exception as e:
        print(f"Error in MSL callback: {e}")
        return go.Figure()



# 回调函数 - 温度图
# @app.callback(
#     Output('temp-plot', 'figure'),
#     [Input('file-selector', 'value'),
#      Input('temp-colorscale', 'value')]
# )
# def update_temp_plot(file_name, colorscale):
#     ds = xr.open_dataset(os.path.join(base_dir, file_name))
#     # 2米温度 (°C)
#     t2m_data = ds['temperature_2m'].values - 273.15
#     fig = go.Figure()
    
#     # 添加温度热力图
#     heatmap = go.Heatmap(
#         x=ds.longitude,
#         y=ds.latitude,
#         z=t2m_data,
#         colorscale=colorscale,
#         colorbar=dict(title='°C'),
#         hovertemplate='温度: %{z:.2f} °C<br>经度: %{x:.2f}°<br>纬度: %{y:.2f}°<extra></extra>'
#     )
    
#     fig.add_trace(heatmap)
#     # fig.add_trace(coastline)
    

#     fig.add_trace(go.Scatter(
#         x=[121.47],
#         y=[31.23],
#         mode='markers+text',
#         marker=dict(size=10, color='red'),
#         text=["上海"],
#         textposition="top center",
#         name="Shanghai",
#         hoverinfo='skip'
#     ))
#     fig.add_trace(go.Scatter(
#                 x=[116.4074],
#                 y=[39.9042],
#                 mode='markers+text',
#                 marker=dict(size=10, color='blue'),
#                 text=["北京"],
#                 textposition="top center",
#                 name="Beijing",
#                 hoverinfo='skip'
#             ))
    

#     timestamp = file_name.replace("output_surface_", "").replace(".nc", "")
#     timestamp = timestamp.replace("-", " ", 3).replace("-", ":")
#     fig.update_layout(
#         title=f"2米温度分布 - {timestamp}",
#         # mapbox_style="carto-positron",
#         # mapbox_zoom=1,
#         # mapbox_center={"lat": 0, "lon": 0},
#         xaxis=dict(
#             title="经度 (°)",
#             showgrid=True,
#             dtick=30,
#             ticks="outside",
#             tick0=0
#         ),
#         yaxis=dict(
#             title="纬度 (°)",
#             showgrid=True,
#             dtick=30,
#             ticks="outside",
#             tick0=-90
#         ),
#         height=600,
#         margin=dict(l=0, r=0, t=50, b=0)
#     )
    
#     return fig
@app.callback(
    Output('temp-plot', 'figure'),
    [Input('file-selector', 'value'), Input('temp-colorscale', 'value')]
)
def update_temp_plot(file_name, colorscale):
    try:
        file_path = os.path.join(base_dir, file_name)
        with xr.open_dataset(file_path) as ds:
            t2m_data = ds['temperature_2m'].values - 273.15
            lon = ds.longitude.values
            lat = ds.latitude.values

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=lon,
            y=lat,
            z=t2m_data,
            colorscale=colorscale,
            colorbar=dict(title='°C'),
            hovertemplate='温度: %{z:.2f} °C<br>经度: %{x:.2f}°<br>纬度: %{y:.2f}°<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=[121.47], y=[31.23], mode='markers+text',
            marker=dict(size=10, color='red'),
            text=["上海"], textposition="top center", name="Shanghai"
        ))
        fig.add_trace(go.Scatter(
            x=[116.4074], y=[39.9042], mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=["北京"], textposition="top center", name="Beijing"
        ))

        timestamp = file_name.replace("output_surface_", "").replace(".nc", "").replace("-", " ", 3).replace("-", ":")

        fig.update_layout(
            title=f"2米温度分布 - {timestamp}",
            xaxis=dict(title="经度 (°)", showgrid=True, dtick=30),
            yaxis=dict(title="纬度 (°)", showgrid=True, dtick=30),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig
    except Exception as e:
        print(f"Error in temperature callback: {e}")
        return go.Figure()



# 回调函数 - 风场图
# @app.callback(
#     Output('wind-plot', 'figure'),
#     [Input('file-selector', 'value'),
#      Input('wind-downsample', 'value')]
# )
# def update_wind_plot(file_name, skip):
#     try:
#         with xr.open_dataset(os.path.join(base_dir, file_name)) as ds:
#             # 10米风场
#             u10_data = ds['u_component_of_wind_10m'].values
#             v10_data = ds['v_component_of_wind_10m'].values

#             # 数据采样
#             u10 = u10_data[::skip, ::skip]
#             v10 = v10_data[::skip, ::skip]
#             speed = np.sqrt(u10**2 + v10**2)

#             lon = ds.longitude[::skip]
#             lat = ds.latitude[::skip]

#             # 创建风速热力图
#             fig = go.Figure(data=go.Heatmap(
#                 z=speed,
#                 x=lon,
#                 y=lat,
#                 colorscale='Viridis',
#                 colorbar=dict(title='风速 m/s')
#             ))

#             # 添加上海点标记
#             fig.add_trace(go.Scatter(
#                 x=[121.47],
#                 y=[31.23],
#                 mode='markers+text',
#                 marker=dict(size=10, color='red'),
#                 text=["上海"],
#                 textposition="top center",
#                 name="Shanghai",
#                 hoverinfo='skip'
#             ))

#             timestamp = file_name.replace("output_surface_", "").replace(".nc", "")
#             timestamp = timestamp.replace("-", " ", 3).replace("-", ":")

#             fig.update_layout(
#                 title=f"10米风速热图 - {timestamp}",
#                 xaxis_title="经度",
#                 yaxis_title="纬度",
#                 height=600
#             )

#             return fig

#     except Exception as e:
#         print(f"Error in wind plot callback: {e}")
#         return go.Figure()  # 如果出错，返回空图形

@app.callback(
    Output('wind-plot', 'figure'),
    [Input('file-selector', 'value'), Input('wind-downsample', 'value')]
)
def update_wind_plot(file_name, skip):
    try:
        file_path = os.path.join(base_dir, file_name)
        with xr.open_dataset(file_path) as ds:
            u10_data = ds['u_component_of_wind_10m'].values
            v10_data = ds['v_component_of_wind_10m'].values
            lon = ds.longitude.values[::skip]
            lat = ds.latitude.values[::skip]

        u10 = u10_data[::skip, ::skip]
        v10 = v10_data[::skip, ::skip]
        speed = np.sqrt(u10**2 + v10**2)

        fig = go.Figure(data=go.Heatmap(
            z=speed, x=lon, y=lat,
            colorscale='Viridis',
            colorbar=dict(title='风速 m/s')
        ))

        fig.add_trace(go.Scatter(
            x=[121.47], y=[31.23],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=["上海"],
            textposition="top center",
            name="Shanghai",
            hoverinfo='skip'
        ))

        timestamp = file_name.replace("output_surface_", "").replace(".nc", "").replace("-", " ", 3).replace("-", ":")

        fig.update_layout(
            title=f"10米风速热图 - {timestamp}",
            xaxis_title="经度",
            yaxis_title="纬度",
            height=600
        )

        return fig
    except Exception as e:
        print(f"Error in wind plot callback: {e}")
        return go.Figure()


if __name__ == '__main__':
    app.run(debug=True)