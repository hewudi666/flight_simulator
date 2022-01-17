import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, inf
from task5 import flight_simulator

def dist(lng1, lat1, lng2, lat2):
    """
    :param lng1:
    :param lat1:
    :param lng2:
    :param lat2:
    :return: 距离，单位是km
    """
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 5)  # 保留两位小数

    return distance

# def duichong():
#     flight_num = 332
#     dp_time = []
#     for i in range(flight_num):
#         dp_time.append(0)
#     path_select = []
#     for i in range(flight_num):
#         path_select.append(0)
#
#     simulator = flight_simulator(dp_time, path_select)
#     duichong_id = []
#
#     for i in range(flight_num):
#         duichong = []
#         depart = simulator.depart_port[i]
#         arrival = simulator.arrival_port[i]
#         for j in range(flight_num):
#             DP = simulator.depart_port[j]
#             AR = simulator.arrival_port[j]
#             if depart == AR and arrival == DP:
#                 duichong.append(i)
#                 duichong.append(j)
#             else:
#                 continue
#         if duichong and duichong[::-1] not in duichong_id:
#             duichong_id.append(duichong)
#
#     return duichong_id


def main():
    flight_num = 953
    dp_time = []
    for i in range(flight_num):
        dp_time.append(0)
    path_select = []
    for i in range(flight_num):
        path_select.append(0)

    simulator = flight_simulator(dp_time, path_select)
    # 航班数量 * 最大航路可选择数量
    path_len_total = np.zeros((simulator.flight_num, max(simulator.path_num)))

    for i in range(simulator.flight_num):
        for j in range(simulator.path_num[i]):
            length = 0
            for k in range(simulator.route_segment_total[i][j]):
                path = simulator.route_total[i][j]
                point1 = path[k]
                point2 = path[k + 1]
                d = dist(point1[0], point1[1], point2[0], point2[1])
                length += d
            path_len_total[i][j] = length

    np.save("path_length_953.npy", path_len_total)


if __name__ == '__main__':
    main()
    # conflict = duichong()
    # print("conflict:", conflict)
    # a = np.array(conflict)
    # print(a)