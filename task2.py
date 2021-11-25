"""
任务2：飞行模拟器的构建
"""

import pandas as pd
import numpy as np
import random
import math
import os
import copy
from math import radians, cos, sin, asin, sqrt, inf
import matplotlib.pyplot as plt

class flight_simulator:

    def __init__(self, flight_data, dp_port, ar_port):
        """
        :param flight_data: 飞行数据
        :param dp_port: 起飞机场
        :param ar_port: 降落机场
        """
        self.flight_data = flight_data
        self.dp_port = dp_port
        self.ar_port = ar_port
        self.flight_time_total = [] # 所有飞行器飞行时刻---相应值为1，状态为飞行；相应值为0，状态为停滞
        # 区域边界 经度75-135，纬度15-55
        self.row = (75, 135)
        self.col = (15, 55)
        # 经纬度每度合100km
        self.angle_km = 100
        # 速度范围: 0.14km/s-0.23km/s
        self.speed_range = (0.14, 0.23)
        self.flight_num = 100  # 飞行器数量
        self.start = 0  # 起始序号
        self.safe_r = 9.26  # 安全半径/km
        self.flight_pos_total = []  # 记录每个时刻所有飞行器的位置
        self.direction_total = []  # 航向角
        self.conflict_num = 0  # 冲突总数
        self.t_step = 10  # 时间步长/s
        self.T = 1000  # 总时间--T个时间步长

        self.route_total = []
        for i in range(len(self.flight_data)):
            route = self.flight_data[i]
            route = self.data_process(route)
            self.route_total.append(route)

        # 飞行器的状态---是否到达终点，到达则为1，否则为0
        self.teriminal = []
        for i in range(self.flight_num):
            self.teriminal.append(0)

        # 生成起飞机场和降落机场初始化
        self.depart_port = []
        self.arrival_port = []

        for i in range(len(self.dp_port)):
            a = self.dp_port[i][1:-1]
            b = self.ar_port[i][1:-1]
            a = a.replace(',', '')
            b = b.replace(',', '')
            a = a.split()
            b = b.split()
            lon1 = float(a[0])
            lat1 = float(a[1])
            lon2 = float(b[0])
            lat2 = float(b[1])
            self.depart_port.append((lon1, lat1))
            self.arrival_port.append((lon2, lat2))

        for i in range(len(self.route_total)):
            r = []
            s = self.route_total[i]
            l = len(s)
            for j in range(int(l / 2)):
                x = s[2 * j]
                y = s[2 * j + 1]
                r.append((x, y))
            self.route_total[i] = r

        self.depart_total = self.depart_port[self.start:self.start + self.flight_num]
        self.arrival_total = self.arrival_port[self.start:self.start + self.flight_num]
        self.flight_path_total = self.route_total[self.start:self.start + self.flight_num]
        self.flight_id_total = []
        self.flight_pos_total.append(self.depart_total)

        # 速度初始化
        self.velocity = (self.speed_range[1] - self.speed_range[0]) * np.random.random((1, self.flight_num)) + self.speed_range[0]
        #print("各飞行器速度：km/s", self.velocity[0])
        self.velocity = self.velocity[0] / self.angle_km

        # 航向角初始化
        for i in range(self.flight_num):
            direct = []
            flight_id = []
            flight_path = self.flight_path_total[i]
            for j in range(len(flight_path) - 1):
                dp = flight_path[j]
                ar = flight_path[j + 1]
                angle = self.angle_compute(dp[0], dp[1], ar[0], ar[1])
                direct.append(angle)
                flight_id.append(j)
            self.direction_total.append(direct)
            self.flight_id_total.append(flight_id)

        # 航路段初始化
        self.current_id = []
        for i in range(self.flight_num):
            self.current_id.append(0)

        # 飞行时刻设定
        for i in range(self.flight_num):
            flight_time = []
            for j in range(self.T):
                flight_time.append(1)
            self.flight_time_total.append(flight_time)


    """
    	目标1：数据进行图结构化（即构建实际航路网络）
    根据现有数据，构建一个图矩阵，其中，图的节点是航路点（包括机场），图的连边代表节点间的距离，有三种取值：
    1）无穷大，代表两航路点之间无连边；
    2）0，航路点与自身航路点间距离为0；
    3）实际数值，根据数据中两航路点之间经纬坐标计算得出。

    把每一条路径的航路点遍历一遍，构建字典和航路点列表，包括航路点的编号，上一个点编号，经纬度坐标
    字典示例： dict = {'id': , 'pos': , 'last': []}
    第一条路径顺序添加即可
    对于后面的每条路径，依次将航路点字典添加
    添加的过程中注意：
    判断航路点是否已添加：
    如果已经添加，需要扩充last对应的列表；如果没有添加，则构建相应字典添加进去，并把坐标元胞添加到航路点列表
    """

    # 去掉'[]','()',','符号
    def data_process(self, route):
        route = route.replace('[', '')
        route = route.replace(']', '')
        route = route.replace('(', '')
        route = route.replace(')', '')
        route = route.replace(',', '')
        route = route.split()
        for i in range(len(route)):
            route[i] = float(route[i])

        return route

    def route_dict(self):
        """
        获取航路点之间的关系
        :param route_total:
        :return:
        """
        route_dict = []
        flight_point = []
        id = 0
        id_port = []

        for Route in self.route_total:
            length = len(Route)
            for i in range(length):
                point = Route[i]
                if point in flight_point:
                    index = flight_point.index(point)
                    """
                    扩充last对应的列表
                    """
                    if i == 0:
                        continue
                    else:
                        Last_point = Route[i - 1]
                        Index1 = flight_point.index(Last_point)
                        if Index1 in route_dict[index]['last']:
                            continue
                        else:
                            route_dict[index]['last'].append(Index1)
                else:
                    """
                    构建相应字典添加进去，并把坐标元胞添加到航路点列表
                    """
                    flight_point.append(point)
                    if i == 0:
                        """
                        如果是路径第一个点，last列表置空
                        """
                        dict = {'id': id, 'pos': point, 'last': []}
                    else:
                        last_point = Route[i - 1]
                        Index = flight_point.index(last_point)
                        dict = {'id': id, 'pos': point, 'last': [Index]}
                    id += 1
                    route_dict.append(dict)

        for i in range(len(flight_point)):
            p = flight_point[i]
            if p in self.depart_port or p in self.arrival_port:
                id_port.append(i)

        return route_dict, flight_point, id_port

    def dist_id(self, flight_point, id1, id2):
        """
        根据两点索引号计算之间距离
        :param flight_point:
        :param id1:
        :param id2:
        :return: 两点之间的距离
        """
        point1 = flight_point[id1]
        point2 = flight_point[id2]
        lng1 = point1[0]
        lat1 = point1[1]
        lng2 = point2[0]
        lat2 = point2[1]
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
        distance = round(distance / 1000, 2)  # 保留两位小数

        return distance

    def graph_matrix(self):
        """
        计算图矩阵
        :return: 图矩阵，邻接矩阵，连边数量，机场id
        """

        route_dict, flight_point, id_port = self.route_dict()

        num = len(flight_point)
        graph_matrix = np.full((num, num), inf)
        graph_matrix1 = np.zeros((num,num)) #图的邻接矩阵
        row, col = np.diag_indices_from(graph_matrix)
        graph_matrix[row, col] = 0
        edge_num = 0  # 存在连边的数量

        for i in range(num):
            p = route_dict[i]
            id1 = p['id']
            last = p['last']
            for loc in last:
                distance = self.dist_id(flight_point, id1, loc)
                graph_matrix[loc][i] = distance
                graph_matrix1[loc][i] = 1
                edge_num += 1

        return graph_matrix, graph_matrix1, edge_num, id_port

    # def find_all_path(self, graph, start, end, path = []):
    #     path = path + [start]
    #     if start == end:
    #         return [path]
    #
    #     path_all = []
    #     index = []
    #     for i in range(len(graph[start])):
    #         if graph[start][i] == 1:
    #             index.append(i)
    #     for node in index:
    #         if node not in path:
    #             s = self.find_all_path(graph, node, end, path)
    #             for n in s:
    #                 path_all.append(n)
    #
    #     return path_all

    def find_path(self, dp_port, ar_port):
        path_all = []
        if dp_port in self.depart_port and ar_port in self.arrival_port:
            for i in range(len(self.dp_port)):
                if dp_port == self.depart_port[i] and ar_port == self.arrival_port[i]:
                    if self.route_total[i] not in path_all:
                        path_all.append(self.route_total[i])
                        # print(self.route_total[i])
                    else:
                        continue

            if not path_all:
                print("没有找到路径！")

            else:
                print("找到路径！")
                return path_all

        else:
            print("没有找到路径！")



    """
    	目标2：飞行过程模拟计算
    随机生成一批飞行器，并初始化飞行器起飞机场、目标机场、飞行器速度（假设匀速飞行）。
    使用线性外推法，计算所有飞行器在每一时刻所在的位置，进一步可以计算在整个运行过程中，所有飞机的冲突总次数。
    要求此飞行模拟器可以实现：
    1）可以输出指定时刻t（0<t<T）时，所有飞机在航路网上的位置；
    2）0-T时段内，整个航路网络上飞行器冲突总次数（给定一个安全距离）。

    区域划定：
    lon 75~135 , lat 15~55  (60×40)
    flight_num架飞机，各自初始化起飞，目标点
    根据起飞-目标点计算速度方向，每个时间步更新位置，判断与目标点的距离，判断与其他飞机的距离
    """

    @staticmethod
    def angle_compute(x1, y1, x2, y2):
        """
        计算方位角函数
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return: 航向角
        """
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        angle = 0
        dy = y2 - y1
        dx = x2 - x1
        if dx == 0 and dy > 0:
            angle = 0
        if dx == 0 and dy < 0:
            angle = 180
        if dy == 0 and dx > 0:
            angle = 90
        if dy == 0 and dx < 0:
            angle = 270
        if dx > 0 and dy > 0:
            angle = math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy > 0:
            angle = 360 + math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        elif dx > 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        return angle * math.pi / 180

    @staticmethod
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
        distance = round(distance / 1000, 2)  # 保留两位小数

        return distance


    def d_compute(self, flight_pos, current_pos):
        """
        :param flight_pos: 当前时刻每架飞行器的位置
        :param current_pos: 当前时刻当前飞行器的位置
        :return: 当前飞行器到其他飞行器的距离
        """
        distance = []
        lon1 = current_pos[0]
        lat1 = current_pos[1]
        # index = flight_pos.index(current_pos)
        for i in range(len(flight_pos)):
            pos = flight_pos[i]
            distance.append(self.dist(lon1, lat1, pos[0], pos[1]))

        return distance


    def flight_update(self):
        # 每个时刻更新每个飞行器的位置，判断是否冲突
        for t in range(1, self.T):
            flight_pos = []
            current_pos = self.flight_pos_total[t - 1]
            # 所有飞机均已到达目标点
            if sum(self.teriminal) == self.flight_num:
                print("所有飞行器已到达目标点！")
                break
            for i in range(self.flight_num):
                if self.flight_time_total[i][t] == 1:
                    current_flight_path = self.flight_path_total[i]
                    k = self.current_id[i]  # 当前航路段
                    depart = current_flight_path[k]
                    arrival = current_flight_path[k + 1]
                    direction = self.direction_total[i]
                    """
                    更新每个飞行器位置和状态    
                    """
                    # 如果已经到达终点，则位置不变
                    if self.teriminal[i] == 1:
                        flight_pos.append(current_pos[i])
                    else:
                        """
                        遍历每个航路段，更新当前航向角，更新位置
                        """
                        # 到目标点的距离
                        d = self.dist(current_pos[i][0], current_pos[i][1], arrival[0], arrival[1])
                        # 如果距离大于一个步长，则线性外推更新位置
                        step = self.t_step * self.velocity[i] * self.angle_km  # 每个时间步走4~6km
                        if d > step:
                            lon = current_pos[i][0] + self.t_step * self.velocity[i] * cos(direction[k])
                            lat = current_pos[i][1] + self.t_step * self.velocity[i] * sin(direction[k])
                            flight_pos.append((lon, lat))
                        else:
                            if self.current_id[i] == self.flight_id_total[i][-1]:
                                flight_pos.append(self.arrival_total[i])
                                self.teriminal[i] = 1
                            else:
                                flight_pos.append(arrival)
                                self.current_id[i] += 1
                else:
                    continue

                print("当前时刻{}，飞行器{}的位置为{}".format(t, i, flight_pos[i]))

            self.flight_pos_total.append(flight_pos)
            """
            判断当前时刻是否冲突，统计冲突数量
            """
            for i in range(self.flight_num):
                dist_t = self.d_compute(self.flight_pos_total[t], self.flight_pos_total[t][i])
                for j in range(len(dist_t)):
                    if dist_t[j] != 0 and dist_t[j] <= self.safe_r:
                        # 如果二者中任何一个已经到达目标点，则跳过
                        if self.teriminal[i] == 1 or self.teriminal[j] == 1:
                            continue
                        else:
                            self.conflict_num += 1


    def dist_target(self):
        """
        计算结束时各飞行器到目标点的距离
        :return:
        """
        dist_target = []
        for i in range(self.flight_num):
            pos = self.flight_pos_total[-1][i]
            target = self.arrival_total[i]
            D = self.dist(pos[0], pos[1], target[0], target[1])
            dist_target.append(D)

        return dist_target

def main():

    print("flight_simulator start!")
    path = "C:\\Users\\lenovo\\PycharmProjects\\project\\处理后飞行数据\\"
    path_list = os.listdir(path)
    # 20191001的飞行数据
    path_main = path_list[92]
    path_all = []
    df = pd.read_excel(path + path_main)
    flight_data_main = df['飞行路径']
    dp_port_main = df['起飞机场坐标']
    ar_port_main = df['降落机场坐标']
    simulator_main = flight_simulator(flight_data_main, dp_port_main, ar_port_main)

    depart = (121.36, 37.4)
    arrival = (114.2, 30.81)
    for i in range(len(path_list)):
        data = pd.read_excel(path + path_list[i])
        flight_data = data['飞行路径']
        dp_port = data['起飞机场坐标']
        ar_port = data['降落机场坐标']

        simulator = flight_simulator(flight_data, dp_port, ar_port)

        path = simulator.find_path(depart, arrival)

        for route in path:
            if route not in path_all:
                path_all.append(route)
            else:
                continue

        # graph_matrix, graph_matrix1, edge_num, id_port = simulator.graph_matrix()
        #
        # print("图矩阵：", graph_matrix)
        # print("机场编号：", id_port)
        # print("连边数量：", edge_num)


    print("起飞机场{}到降落机场{}之间的所有路径为：".format(depart, arrival))
    print(path_all)

    # simulator.flight_update()
    # dist_target = simulator.dist_target()
    #
    # print('起始点位置：', simulator.depart_total)
    # print('目标点位置：', simulator.arrival_total)
    # print("最后时刻各飞行器离目标点的距离：/km", dist_target)
    # print("所有飞行器状态：", simulator.teriminal)
    # print("所有飞行器当前所在航路段：", simulator.current_id)
    # print("最后时刻所有飞行器位置：", simulator.flight_pos_total[-1])
    # print("冲突总数：", simulator.conflict_num)
    #
    # """
    # 飞行器运行轨迹
    # """
    #
    # lon_total = []
    # lat_total = []
    # for i in range(simulator.flight_num):
    #     x = []
    #     y = []
    #     for j in range(len(simulator.flight_pos_total)):
    #         p = simulator.flight_pos_total[j][i]
    #         x.append(p[0])
    #         y.append(p[1])
    #     # x.append(arrival_total[i][0])
    #     # y.append(arrival_total[i][1])
    #     lon_total.append(x)
    #     lat_total.append(y)
    #     plt.plot(x, y)
    #
    # plt.show()


if __name__ == '__main__':
    main()
