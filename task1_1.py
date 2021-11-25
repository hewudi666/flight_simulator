"""
根据处理后的飞行数据绘图验证
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#去掉'[]','()',','符号
def data_process(route):
    route = route.replace('[', '')
    route = route.replace(']', '')
    route = route.replace('(', '')
    route = route.replace(')', '')
    route = route.replace(',', '')
    route = route.split()
    for i in range(len(route)):
        route[i] = float(route[i])

    return route

data = pd.read_csv('飞行数据_处理后.csv')
flight_data = data['飞行路径']
route_total = []
lon_total = []
lat_total = []
for i in range(len(flight_data)):
    route = flight_data[i]
    route = data_process(route)
    route_total.append(route)
    print(route)
    lon = route[0::2]
    lat = route[1::2]
    lon_total.append(lon)
    lat_total.append(lat)

for i in range(len(flight_data)):
    plt.plot(lon_total[i], lat_total[i])

plt.show()








