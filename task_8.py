import pandas as pd
from task2 import flight_simulator
import time

start = time.time()
df = pd.read_excel("C:\\Users\\lenovo\\PycharmProjects\\project\\flight_simulator\\处理后飞行数据\\20190901飞行数据_处理后.xls")
dp_port = df['起飞机场坐标']
ar_port = df['降落机场坐标']
flight_data = df['飞行路径']
dp_time = df['起飞时间']
ar_time = df['降落时间']
simulator = flight_simulator(flight_data, dp_port, ar_port)
# print(dp_time[3][-5:])
# print(dp_time[0])


data = pd.read_excel("find_all_path.xls")
dp_total = data['起飞机场坐标']
ar_total = data['降落机场坐标']
dp_t = data['起飞时间']
ar_t = data['降落时间']
path_num = data['可选路径个数']
flight_route = data['飞行路径']

a = dp_total.values.tolist()
b = dp_t.values.tolist()
c = path_num.values.tolist()
print(a)
print(type(c[0]))


def data_process1(route):
    route = route.replace('[', '')
    route = route.replace(']', '')
    route = route.replace('(', '')
    route = route.replace(')', '')
    route = route.replace(',', '')
    route = route.split()
    for i in range(len(route)):
        route[i] = float(route[i])

    return route

def data_process2(route, depart, arrival):
    route_new = []
    r = []
    for i in range(len(route)):
        pos = route[i]
        if pos == depart:
            r = []
            r.append(pos)
        elif pos == arrival:
            r.append(pos)
            route_new.append(r)
        else:
            r.append(pos)
    return route_new


def data_process(data):
    dp_total = data['起飞机场坐标']
    ar_total = data['降落机场坐标']
    dp_t = data['起飞时间']
    ar_t = data['降落时间']
    path_num = data['可选路径个数']
    flight_route = data['飞行路径']

    dp_port_total = []
    ar_port_total = []
    dp_time = dp_t.values.tolist()
    ar_time = ar_t.values.tolist()
    path_n = path_num.values.tolist()
    route_total = []

    num = len(dp_total)

    for i in range(num):
        a = dp_total[i][1:-1]
        b = ar_total[i][1:-1]
        a = a.replace(',', '')
        b = b.replace(',', '')
        a = a.split()
        b = b.split()
        lon1 = float(a[0])
        lat1 = float(a[1])
        lon2 = float(b[0])
        lat2 = float(b[1])
        dp_port_total.append((lon1, lat1))
        ar_port_total.append((lon2, lat2))

    r_total = []
    for i in range(num):
        route = flight_route[i]
        route = data_process1(route)
        r_total.append(route)
    for i in range(len(r_total)):
        r = []
        s = r_total[i]
        l = len(s)
        for j in range(int(l / 2)):
            x = s[2 * j]
            y = s[2 * j + 1]
            r.append((x, y))
        r_total[i] = r

    # 飞行路径列表
    for i in range(num):
        route = r_total[i]
        dp = dp_port_total[i]
        ar = ar_port_total[i]
        route_new = data_process2(route, dp, ar)
        route_total.append(route_new)

    return dp_port_total, ar_port_total, dp_time, ar_time, path_n, route_total


depart_port, arrival_port, depart_time, arrival_time, path_n, flight_route = data_process(data)

print("depart_port:", depart_port)
print("arrival_port:", arrival_port)
print("depart_time:", depart_time)
print("arrival_time:", arrival_time)
print("path_n:", path_n)
print("flight_route:", len(flight_route))

qifei_port = []
jiangluo_port = []
qifei_t = []
jiangluo_t = []
path_len = []
feixing_route = []
data_save = pd.DataFrame()

num = len(depart_port)
for i in range(num):
    dp = depart_port[i]
    ar = arrival_port[i]
    d_t = depart_time[i]
    a_t = arrival_time[i]
    p_l = path_n[i]
    r_l = flight_route[i]

    qifei_port.append(dp)
    jiangluo_port.append(ar)
    qifei_t.append(d_t)
    jiangluo_t.append(a_t)
    path_len.append(p_l)
    feixing_route.append(r_l)

    for j in range(len(simulator.depart_port)):
        print("{}__{}processing".format(i, j))
        if dp == simulator.depart_port[j] and ar == simulator.arrival_port[j]:
            if dp_time[j] == d_t and ar_time[j] == a_t:
                continue
            else:
                qifei_port.append(dp)
                jiangluo_port.append(ar)
                qifei_t.append(dp_time[j])
                jiangluo_t.append(ar_time[j])
                path_len.append(p_l)
                feixing_route.append(r_l)


data_save['起飞机场坐标'] = qifei_port
data_save['降落机场坐标'] = jiangluo_port
data_save['起飞时间'] = qifei_t
data_save['降落时间'] = jiangluo_t
data_save['可选路径个数'] = path_len
data_save['飞行路径'] = feixing_route
end = time.time()
print("用时 {} s".format(end - start))
data_save.to_excel("data_large_scale.xls")





















