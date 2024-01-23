from random import randrange
import json
import glob
import datetime
import os
import joblib
import random
from sklearn.preprocessing import MinMaxScaler

import pandas as pd


from flask import Flask, render_template, send_file, make_response, request
from flask.json import jsonify

from pyecharts import options as opts
from pyecharts.charts import Line, Grid

app = Flask(__name__, static_folder="static")

raw_data = pd.read_csv('data/weekly_time_series.csv')
model = joblib.load('data/model.pkl')
customer_id = 'KHBM00001'
xData = []
yData = []
rndCustomer = {'客户唯一编码': [], 
               '数据更新日期': [], 
               '登录率': [], 
               '客户活跃账户数': [], 
               'BPM新增实例数': [], 
               'CRM新建实例数': [], 
               '修改客户数': [],
               '是否流失': [], 
               '是否流失(提前预测)': [], 
            #    '标准化登录率': [], 
            #    '标准化客户活跃账户数': [], 
            #    '标准化BPM新增实例数': [],
            #    '标准化CRM新建实例数': [], 
            #    '标准化修改客户数': [], 
            #    '标准化登录率_变化': [], 
            #    '标准化客户活跃账户数_变化': [],
            #    '标准化BPM新增实例数_变化': [], 
            #    '标准化CRM新建实例数_变化': [], 
            #    '标准化修改客户数_变化': [], 
            #    '登录率_变化累计': [], 
            #    '登录率波动性': [],
            #    '模型预测结果': []
               }

@app.route("/")
def index():
    return render_template("demo.html")


@app.route("/initChart")
def init_chart():
    print('initChart')
    l1 = (
        Line()
        .add_xaxis(
            xaxis_data=[1],
        )
        .add_yaxis(
            series_name="acc",
            y_axis=[1],
            label_opts=opts.LabelOpts(is_show=False),
            is_step=True,
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
            ),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True, link=[{"xAxisIndex": "all"}],
                
            ),
    #         datazoom_opts=[
    #             opts.DataZoomOpts(
    #                 is_show=True,
    #                 is_realtime=True,
    #                 start_value=30,
    #                 end_value=70,
    #                 xaxis_index=[0, 1],
    #             )
    #         ],
            xaxis_opts=opts.AxisOpts(
                type_="category", 
                boundary_gap=False,
                axislabel_opts = opts.LabelOpts(is_show=False),
                is_show=False,
            ),
            yaxis_opts=opts.AxisOpts(
                axispointer_opts=opts.AxisPointerOpts(
                    label = opts.LabelOpts(is_show=False),
                ),
                is_show=False,
            ),
            legend_opts=opts.LegendOpts(
                pos_left="left",
                is_show=False,
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=True,
                    is_realtime=True,
                    type_="inside",
                    start_value=0,
                    end_value=100,
                    xaxis_index=[0],
                ),
                opts.DataZoomOpts(
                    is_show=False,
                    is_realtime=True,
                    type_="slider",
                    start_value=0,
                    end_value=100,
                    xaxis_index=[0],
                    pos_top=30,
                )
            ],
        )
    )

    grid = Grid(init_opts=opts.InitOpts(width="1000px", height="1000px"))
    grid.add(chart=l1, grid_opts=opts.GridOpts(height=1))
    # grid.add(chart=l2, grid_opts=opts.GridOpts(pos_left=50, pos_right=50, pos_top=200, height=100))
    # grid.add(chart=l3, grid_opts=opts.GridOpts(pos_left=50, pos_right=50, pos_top=250, height=100))

    return grid.dump_options_with_quotes()


@app.route("/addSignal/<signal_name>/")
def addSignal(signal_name):
    global customer_id, xData
    sigData = list(raw_data[raw_data['客户唯一编码'] == customer_id][signal_name])
    print("Add signal: %s"%signal_name)

    series_data = []
    print('len xData:%s'%len(xData))
    for i in range(len(xData)):
        series_data.append([xData[i], sigData[i]])

    # if signal_name in data_dict.keys():
    #     series_data = []
    #     print(len(xData), xData[:10])
    #     print(len(data_dict[signal_name]),data_dict[signal_name][:10])
    #     for i in range(len(xData)):
            
    #     # for i in range(500):
    #         dt = datetime.datetime.fromtimestamp(xData[i] / 1e9)
    #         series_data.append([dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], data_dict[signal_name][i]])

    return jsonify({
        "result": "ok",
        "signal_name": signal_name, 
        "series_data": series_data,
    })



@app.route("/setCustomer/<c_id>/")
def setCustomer(c_id):
    global customer_id, xData
    customer_id = c_id

    print('Set customer_id as %s'%c_id)

    xData = list(raw_data[raw_data['客户唯一编码'] == customer_id]['数据更新日期'])

    # if signal_name in data_dict.keys():
    #     series_data = []
    #     print(len(xData), xData[:10])
    #     print(len(data_dict[signal_name]),data_dict[signal_name][:10])
    #     for i in range(len(xData)):
            
    #     # for i in range(500):
    #         dt = datetime.datetime.fromtimestamp(xData[i] / 1e9)
    #         series_data.append([dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], data_dict[signal_name][i]])

    return jsonify({
        "result": "ok",
        "customer_id": customer_id, 
    })


@app.route("/randomData")
def randomData():
    global rndCustomer, raw_data
    import pandas as pd


    # 获取当前日期
    current_date = pd.to_datetime('today').strftime('%Y-%m-%d')

    # 增加一个月
    new_date = pd.to_datetime('today') + pd.DateOffset(months=len(rndCustomer['登录率']))

    print(new_date)
    if len(rndCustomer['登录率']) == 0:
        rndCustomer['客户唯一编码'].append("TEST")
        rndCustomer['数据更新日期'].append(current_date)
        rndCustomer['登录率'].append(random.random())
        rndCustomer['客户活跃账户数'].append(0.0)
        rndCustomer['BPM新增实例数'].append(0.0)
        rndCustomer['CRM新建实例数'].append(0.0)
        rndCustomer['修改客户数'].append(0.0)
        rndCustomer['是否流失'].append(0.0)
        rndCustomer['是否流失(提前预测)'].append(0.0)
        
    else:

        rndCustomer['客户唯一编码'].append("TEST")
        rndCustomer['数据更新日期'].append(new_date.strftime('%Y-%m-%d'))
        rndCustomer['登录率'].append(rndCustomer['登录率'][-1]*(1+random.uniform(0.1, 0.9)*random.choice([-1, 1])))
        rndCustomer['客户活跃账户数'].append(0.0)
        rndCustomer['BPM新增实例数'].append(0.0)
        rndCustomer['CRM新建实例数'].append(0.0)
        rndCustomer['修改客户数'].append(0.0)
        rndCustomer['是否流失'].append(0.0)
        rndCustomer['是否流失(提前预测)'].append(0.0)
    
    scaler = MinMaxScaler()

    customer_pd = pd.DataFrame(rndCustomer)
    print(customer_pd)
    customer_pd['标准化登录率'] = scaler.fit_transform(customer_pd['登录率'].values.reshape(-1, 1))
    customer_pd['标准化客户活跃账户数'] = scaler.fit_transform(customer_pd['客户活跃账户数'].values.reshape(-1, 1))
    customer_pd['标准化BPM新增实例数'] = scaler.fit_transform(customer_pd['BPM新增实例数'].values.reshape(-1, 1))
    customer_pd['标准化CRM新建实例数'] = scaler.fit_transform(customer_pd['CRM新建实例数'].values.reshape(-1, 1))
    customer_pd['标准化修改客户数'] = scaler.fit_transform(customer_pd['修改客户数'].values.reshape(-1, 1))

    for col in customer_pd.columns:
        if col.startswith('标准化'):
            customer_pd[f'{col}_变化'] = customer_pd[col].diff()

    customer_pd['登录率_变化累计'] = customer_pd['标准化登录率_变化'].rolling(window=3).sum()
    customer_pd['登录率波动性'] = customer_pd['标准化登录率'].rolling(window=3).std()

    customer_pd.fillna(0, inplace=True)

    raw_data = raw_data[raw_data['客户唯一编码'] != 'TEST']
    raw_data = pd.concat([raw_data, customer_pd], ignore_index=True)

    y = model.predict([customer_pd.iloc[-1].drop(['客户唯一编码', '数据更新日期', '是否流失', '是否流失(提前预测)'])])

    return jsonify({
        "result": "ok",
        "predict": int(y[0]), 
    })

@app.route("/clearData")
def clearData():
    global rndCustomer
    rndCustomer = {'客户唯一编码': [], 
                '数据更新日期': [], 
                '登录率': [], 
                '客户活跃账户数': [], 
                'BPM新增实例数': [], 
                'CRM新建实例数': [], 
                '修改客户数': [],
                '是否流失': [], 
                '是否流失(提前预测)': [], 
                #    '标准化登录率': [], 
                #    '标准化客户活跃账户数': [], 
                #    '标准化BPM新增实例数': [],
                #    '标准化CRM新建实例数': [], 
                #    '标准化修改客户数': [], 
                #    '标准化登录率_变化': [], 
                #    '标准化客户活跃账户数_变化': [],
                #    '标准化BPM新增实例数_变化': [], 
                #    '标准化CRM新建实例数_变化': [], 
                #    '标准化修改客户数_变化': [], 
                #    '登录率_变化累计': [], 
                #    '登录率波动性': [],
                #    '模型预测结果': []
               }

    return jsonify({
        "result": "ok",
    })

@app.route("/predict/<pdata>/")
def predict(pdata):
    global model

    print(pdata)
    y = model.predict([pdata])
    print(y)

    return jsonify({
        "result": "ok",
        "predict": y, 
    })




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001 )
