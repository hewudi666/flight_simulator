"""
任务2：飞行模拟器的构建
"""

import pandas as pd
import numpy as np
import random
import math
import os
import copy
import time
from math import radians, cos, sin, asin, sqrt, inf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'


df = pd.read_excel("find_all_path3_mid.xls")
_flight_data = df['飞行路径']
_dp_port = df['起飞机场坐标']
_ar_port = df['降落机场坐标']


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

def caldis(u,v):
    # 计算输入矩阵中两个向量的距离
    return dist(u[0],u[1],v[0],v[1])

def dist_mat(lon, lat):
    """
    :param lon: 经度矩阵
    :param lat: 维度矩阵
    :return: 距离矩阵
    """
    lon = np.radians(np.array(lon))
    lat = np.radians(np.array(lat))
    N = len(lon)
    lon1 = lon[:, np.newaxis]
    lat1 = lat[:, np.newaxis]
    lon_1 = np.tile(lon1, (1, N))
    lat_1 = np.tile(lat1, (1, N))
    d_lon = lon_1 - lon
    d_lat = lat_1 - lat
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat_1) * np.cos(lat) * np.sin(d_lon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371
    distance = np.around(distance, 2)
    return distance

class flight_simulator:

    def __init__(self, depart_time, path_id, specific_id = []):
        """
        :param depart_time: 起飞时间
        :param path_id: 路径选择
        """
        self.depart_time = depart_time
        self.path_id = path_id
        self.flight_time_total = []  # 所有飞行器飞行时刻---相应值为1，状态为飞行；相应值为0，状态为停滞
        # 经纬度每度合100km
        self.angle_km = 100
        self.flight_num = len(self.depart_time)  # 飞行器数量
        self.start = 0  # 起始序号
        self.safe_r = 9.26  # 安全半径/km
        self.flight_pos_total = []  # 记录每个时刻所有飞行器的位置
        self.direction_total = []  # 航向角
        self.conflict_num = 0  # 冲突总数
        self.t_step = 60  # 时间步长/s
        self.T = 1000  # 总时间--T个时间步长
        self.conflict_num_t = []

        self.arrival_time = np.zeros(self.flight_num) # 各飞行器到达时间
        # 生成起飞机场和降落机场初始化
        self.depart_port = []
        self.arrival_port = []

        if specific_id:
            global flight_data
            global dp_port
            global ar_port
            flight_data = _flight_data[specific_id].reset_index(drop = True)
            dp_port = _dp_port[specific_id].reset_index(drop = True)
            ar_port = _ar_port[specific_id].reset_index(drop = True)
            self.dp_time = df['起飞时间'][specific_id].reset_index(drop = True)
            self.ar_time = df['降落时间'][specific_id].reset_index(drop = True)
            self.path_num = df['可选路径个数'][specific_id].reset_index(drop = True)
        else:
            flight_data = _flight_data
            dp_port = _dp_port
            ar_port = _ar_port
            self.dp_time = df['起飞时间']
            self.ar_time = df['降落时间']
            self.path_num = df['可选路径个数']

        # 最大提前时间
        ahead_max = min(self.depart_time)
        # index = np.argmin(self.depart_time) + self.start
        # if ahead_max + self.dp_time[index] < 0:
        #     self.ahead_of_time = abs(ahead_max + self.dp_time[index])
        # else:
        #     self.ahead_of_time = 0
        if ahead_max < 0:
            self.ahead_of_time = abs(ahead_max)
        else:
            self.ahead_of_time = 0
        # print("最大提前时间", self.ahead_of_time)

        # 延误时间
        self.delay_time = np.zeros(self.flight_num)

        # 飞行器的状态---是否到达终点，到达则为1，否则为0
        self.teriminal = []
        for i in range(self.flight_num):
            self.teriminal.append(0)

        r_total = []
        for i in range(len(flight_data)):
            route = flight_data[i]
            route = self.data_process1(route)
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

        for i in range(len(dp_port)):
            a = dp_port[i][1:-1]
            b = ar_port[i][1:-1]
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

        # 飞行路径列表
        self.route_total = []
        for i in range(len(dp_port)):
            route = r_total[i]
            dp = self.depart_port[i]
            ar = self.arrival_port[i]
            route_new = self.data_process2(route, dp, ar)
            self.route_total.append(route_new)

        route_num_max = []
        route_num_min = []
        for i in range(len(dp_port)):
            a = []
            s = self.route_total[i + self.start]
            for j in range(len(s)):
                num = len(s[j])
                a.append(num)
            max_num = max(a)
            min_num = min(a)
            route_num_max.append(max_num)
            route_num_min.append(min_num)

        self.route_num_max = max(route_num_max)
        # print("route_num_max：", route_num_max)
        # print("最大航路段:", self.route_num_max)
        self.route_num_min = min(route_num_min)
        # print("route_num_min：", route_num_min)
        # print("最小航路段:", self.route_num_min)

        false_id = []
        # 判断输入path_select是否合理
        for i in range(self.flight_num):
            if self.path_id[i] >= self.path_num[i + self.start]:
                false_id.append(i)

        if false_id:
            print("输入path_selection超过列表序号！")
            print("false_id:", false_id)

        # 航路段列表
        self.route_segment_total = []
        for i in range(self.flight_num):
            p = self.route_total[i]
            L = []
            for j in range(self.path_num[i]):
                L.append(len(p[j]) - 1)
            self.route_segment_total.append(L)

        # print("航路段列表:", self.route_segment_total)
        route_segment_all_data_max = []
        for i in range(self.flight_num):
            route_segment_all_data_max.append(max(self.route_segment_total[i]))

        self.route_segment_all_data_max = max(route_segment_all_data_max)
        # print("飞行数据所有路径最大航路段：", self.route_segment_all_data_max)


        self.depart_total = self.depart_port[self.start:self.start + self.flight_num]
        self.arrival_total = self.arrival_port[self.start:self.start + self.flight_num]
        self.flight_path_total = []
        for i in range(self.flight_num):
            self.flight_path_total.append(self.route_total[self.start + i][self.path_id[i]])
        self.flight_id_total = []
        self.flight_pos_total.append(self.depart_total)

        # 计算航路长度
        self.path_len_total = []
        for i in range(self.flight_num):
            R = self.flight_path_total[i]
            path_inter_num = len(R) - 1 # 航路段数量
            length = 0
            for j in range(path_inter_num):
                point1 = R[j]
                point2 = R[j + 1]
                d = self.dist(point1[0], point1[1], point2[0], point2[1]) # 单位：km
                length += d
            self.path_len_total.append(length)

        # print("229号航班路径长度:", self.path_len_total[229])
        self.route_segment = []
        # 寻找最长航路段
        for i in range(self.flight_num):
            path = self.flight_path_total[i]
            self.route_segment.append(len(path) - 1)

        self.route_segment_max = max(self.route_segment)
        # print("航路段长度", self.route_segment)
        # print("route_max:", self.route_segment_max)

        # 速度初始化 speed单位：km/s
        v = np.zeros((self.flight_num, self.route_segment_max))
        for i in range(self.flight_num):
            l = self.path_len_total[i]
            t = (self.ar_time[i + self.start] - self.dp_time[i + self.start]) * 60  # 单位：s
            speed = l / t
            for j in range(self.route_segment[i]):
                v[i][j] = speed

        self.velocity = v
        # print("默认速度控制：", self.velocity)
        # np.save("default_v_953.npy", self.velocity)
        # 航速单位转换 km/s ---> 单位：°/s (经纬度)
        self.velocity = self.velocity / self.angle_km

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

        # 航向角初始化
        self.direction = np.zeros((self.flight_num, self.route_segment_max))
        for i in range(self.flight_num):
            for j in range(len(self.direction_total[i])):
                self.direction[i][j] = self.direction_total[i][j]

        # print("航向角控制:", self.direction)


        # 航路段初始化
        self.current_id = []
        for i in range(self.flight_num):
            self.current_id.append(0)

        # 冲突点
        self.conflict_point = []

        # lon, lat 矩阵
        self.lon_delta = self.t_step * self.velocity * np.cos(self.direction)
        self.lat_delta = self.t_step * self.velocity * np.sin(self.direction)

        # 飞行时刻设定
        for i in range(self.flight_num):
            flight_time = []
            time = self.dp_time[i + self.start] + self.depart_time[i] # 实际起起飞时刻
            # time = self.depart_time[i]  # 实际起起飞时刻
            for j in range(time + self.ahead_of_time):
                flight_time.append(0)

            for k in range(self.T - time - self.ahead_of_time):
                flight_time.append(1)

            self.flight_time_total.append(flight_time)

    def velocity_input(self, velocity_input):
        """
        :param velocity_input: 单位：km/s
        """
        if isinstance(velocity_input, np.ndarray):
            shape = velocity_input.shape
            if shape[0] == self.flight_num and shape[1] >= self.route_segment_max:
                for i in range(self.flight_num):
                    for j in range(self.route_segment_max):
                        self.velocity[i][j] = velocity_input[i][j]
                print("完成速度控制赋值：", self.velocity)
                self.velocity = self.velocity / self.angle_km
                self.lon_delta = self.t_step * self.velocity * np.cos(self.direction)
                self.lat_delta = self.t_step * self.velocity * np.sin(self.direction)
            else:
                print("输入速度控制维度错误!")
        else:
            print("输入速度控制不是numpy数组!")

    # 去掉'[]','()',','符号
    @staticmethod
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

    @staticmethod
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
    @staticmethod
    def caldis(u, v):
        # 计算输入矩阵中两个向量的距离
        return dist(u[0], u[1], v[0], v[1])

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
                # print("所有飞行器已到达目标点！")
                break
            for i in range(self.flight_num):
                if self.flight_time_total[i][t] == 1:
                    current_flight_path = self.flight_path_total[i]
                    k = self.current_id[i]  # 当前航路段
                    depart = current_flight_path[k]
                    arrival = current_flight_path[k + 1]
                    # direction = self.direction_total[i]
                    """
                    更新每个飞行器位置和状态    
                    """
                    # 如果已经到达终点，则位置不变
                    if self.teriminal[i] == 1:
                        flight_pos.append(current_pos[i])
                        # flight_pos.append(self.taibei)
                    else:
                        """
                        遍历每个航路段，更新当前航向角，更新位置
                        """
                        # 到目标点的距离
                        d = self.dist(current_pos[i][0], current_pos[i][1], arrival[0], arrival[1])
                        # 如果距离大于一个步长，则线性外推更新位置
                        step = self.t_step * self.velocity[i][k] * self.angle_km  # 每个时间步走10~12km
                        if d > step:
                            lon = current_pos[i][0] + self.lon_delta[i][k]
                            lat = current_pos[i][1] + self.lat_delta[i][k]
                            flight_pos.append((lon, lat))
                        else:
                            if self.current_id[i] == self.flight_id_total[i][-1]:
                                flight_pos.append(self.arrival_total[i])
                                self.teriminal[i] = 1
                                self.arrival_time[i] = t
                            else:
                                flight_pos.append(arrival)
                                self.current_id[i] += 1
                else:
                    flight_pos.append(self.depart_total[i])

                # print("当前时刻{}，飞行器{}的位置为{}".format(t, i, flight_pos[i]))

            self.flight_pos_total.append(flight_pos)
            """
            判断当前时刻是否冲突，统计冲突数量
            """
            # 方法一
            # for i in range(self.flight_num):
            #     dist_t = self.d_compute(self.flight_pos_total[t], self.flight_pos_total[t][i])
            #     for j in range(len(dist_t)):
            #         if dist_t[j] != 0 and dist_t[j] <= self.safe_r:
            #             # 如果二者中任何一个已经到达目标点，则跳过
            #             if self.teriminal[i] == 1 or self.teriminal[j] == 1:
            #                 continue
            #             else:
            #                 self.conflict_num += 1
            # 方法二
            # d_mat = squareform(pdist(self.flight_pos_total[t], metric=caldis))
            # conflict_num = sum(sum(np.logical_and(d_mat > 0, d_mat <= self.safe_r))) / 2
            # self.conflict_num += conflict_num
            # 方法三
            # a = time.time()
            conflict_p = []
            terminal = np.array(self.teriminal)
            id = np.where(terminal == 1)
            flight_mat = np.array(self.flight_pos_total[t])
            lon_mat = flight_mat[:, 0]
            lat_mat = flight_mat[:, 1]
            d_mat = np.triu(dist_mat(lon_mat, lat_mat), k=0)
            for m in range(len(id)):
                d_mat[id[m], :] = 0
                d_mat[:, id[m]] = 0
            conflict_matrix = np.logical_and(d_mat > 0, d_mat <= self.safe_r)
            # print("conflict_mat", conflict_matrix)
            index = np.where(conflict_matrix == True)
            conflict_num = sum(sum(conflict_matrix))
            # print("confict_num", conflict_num)
            self.conflict_num_t.append(conflict_num)
            self.conflict_num += conflict_num
            # 记录冲突点数
            for j in range(len(index[0])):
                point1 = self.flight_pos_total[t][index[0][j]]
                point2 = self.flight_pos_total[t][index[1][j]]
                if point1 not in conflict_p:
                    conflict_p.append(point1)
                if point2 not in conflict_p:
                    conflict_p.append(point2)

            self.conflict_point.extend(conflict_p)
            # b = time.time()
            # print("########花费时间", b - a)


        # self.conflict_num = self.conflict_num / 2
        # 实际到达时间
        self.arrival_time = self.arrival_time - self.ahead_of_time
        # 计算延误时间 负数代表提前时间，正数为延迟时间
        for i in range(self.flight_num):
            self.delay_time[i] = self.arrival_time[i] - self.ar_time[i + self.start]

        self.delay_time = np.array(self.delay_time).tolist()


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

    def plot_conflict_num(self):
        plt.figure()
        plt.title('冲突频次-时间步')
        plt.xlabel('时间步')
        plt.ylabel('冲突频次')
        x = range(len(self.conflict_num_t))
        plt.plot(x, self.conflict_num_t)
        plt.show()

    def plot(self):
        """
        绘图
        飞行器运行轨迹, 航路点为黄色, 冲突点为红色
        """
        plt.figure(1)
        plt.title('飞行轨迹线路图')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        lon_total = []
        lat_total = []
        for i in range(self.flight_num):
            x = []
            y = []
            for j in range(len(self.flight_pos_total)):
                p = self.flight_pos_total[j][i]
                x.append(p[0])
                y.append(p[1])
            # x.append(arrival_total[i][0])
            # y.append(arrival_total[i][1])
            lon_total.append(x)
            lat_total.append(y)
            # plt.plot(x, y, "b")
            plt.scatter(x, y, s=5, marker='.', c='yellow')

        for i in range(len(self.conflict_point)):
            plt.scatter(self.conflict_point[i][0], self.conflict_point[i][1], s=7, marker='o', c='r')

        plt.show()
        """
        heatmap：
        冲突---1 不冲突----0
        """
        # plt.figure(2)
        # plt.title('飞行轨迹热度图')
        # plt.xlabel('经度')
        # plt.ylabel('纬度')

    def data_save(self):
        route_data = self.flight_pos_total
        file = open('flight_route_data.txt', 'w')
        for fp in route_data:
            file.write(str(fp))
            file.write('\n')
        file.close()


def main():
    flight_num = 925
    dp_time = []
    for i in range(flight_num):
        dp_time.append(0)
    path_select = []
    for i in range(flight_num):
        path_select.append(0)
    # dp_time = np.load('timeslot.npy')
    # path_select = np.load('pathselection.npy')
    # v_control = np.load('default_v.npy')
    # dp_time = np.array(dp_time).astype(int)
    # path_select = np.array(path_select).astype(int) - 1
    # print("dp_time:", dp_time)
    # print("path_select:", path_select)
    # print("v_control:", v_control)
    start = time.time()
    simulator = flight_simulator(dp_time, path_select)
    end1 = time.time()
    print("初始化花费时间:", end1 - start)
    # v_control = v_control / 60
    # simulator.velocity_input(v_control)
    simulator.flight_update()
    end2 = time.time()
    simulator.plot_conflict_num()
    # 更新时间
    print("更新花费时间：", end2 - end1)
    # 冲突次数
    print("冲突次数：", simulator.conflict_num)
    # 飞行器到达状态：0表示未到达, 1表示已到达
    print("到达状态：", simulator.teriminal)
    # 延误时间
    print("延迟时间：", simulator.delay_time)
    # 总延迟时间
    print("总延迟时间:", sum(simulator.delay_time))
    # simulator.data_save()
    # simulator.plot()
    # 最后时刻所有飞行器位置
    # print("最后时刻飞行器位置", simulator.flight_pos_total[-1])


if __name__ == '__main__':
    main()
