import pandas as pd
import os
from task2 import flight_simulator
import time

def main():
    start = time.time()
    print("flight_simulator start!")
    path_dir = "C:\\Users\\lenovo\\PycharmProjects\\project\\flight_simulator\\处理后飞行数据\\"
    path_list = os.listdir(path_dir)
    print("path_list:", path_list)
    # 20190701的飞行数据
    path_main = path_list[62]
    print("path_main:", path_main)
    df = pd.read_excel(path_dir + path_main)
    flight_data_main = df['飞行路径']
    dp_port_main = df['起飞机场坐标']
    ar_port_main = df['降落机场坐标']
    simulator_main = flight_simulator(flight_data_main, dp_port_main, ar_port_main)
    path_all = []
    depart_time = []
    arrival_time = []

    data_save = pd.DataFrame()
    depart_port = []
    arrival_port = []
    dp_ar = []

    for i in range(len(simulator_main.depart_port)):
        find_path = []
        depart = simulator_main.depart_port[i]
        arrival = simulator_main.arrival_port[i]
        if (depart, arrival) in dp_ar:
            continue
        else:
            depart_port.append(depart)
            arrival_port.append(arrival)
            dp_ar.append((depart, arrival))
            for j in range(77, 84):
                print("对第{}对起点终点######进行第{}个文件搜索".format(i, j))
                data = pd.read_excel(path_dir + path_list[j])
                flight_data = data['飞行路径']
                dp_port = data['起飞机场坐标']
                ar_port = data['降落机场坐标']

                simulator = flight_simulator(flight_data, dp_port, ar_port)

                path = simulator.find_path(depart, arrival)

                # 如果能找到路径
                if path:
                    for route in path:
                        if route not in find_path:
                            find_path.append(route)
                        else:
                            continue
                else:
                    continue

        path_all.append(find_path)


    data_save['起飞机场坐标'] = depart_port
    data_save['降落机场坐标'] = arrival_port
    data_save['飞行路径'] = path_all

    end = time.time()
    print("用时 {} s".format(end - start))
    data_save.to_excel("dp_ar_path_all_2.xls")

if __name__ == '__main__':
    main()
