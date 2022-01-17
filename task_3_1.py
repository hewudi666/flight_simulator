import pandas as pd
from task2 import flight_simulator

df = pd.read_excel("C:\\Users\\lenovo\\PycharmProjects\\project\\flight_simulator\\处理后飞行数据\\20190901飞行数据_处理后.xls")
dp_port = df['起飞机场坐标']
ar_port = df['降落机场坐标']
flight_data = df['飞行路径']
dp_time = df['起飞时间']
ar_time = df['降落时间']
simulator = flight_simulator(flight_data, dp_port, ar_port)
print(dp_time[3][-5:])


def data_process(route, depart, arrival):
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


def main():
    path_dir = "C:\\Users\\lenovo\\PycharmProjects\\project\\flight_simulator\\path_0901\\"
    path_list = ['dp_ar_path_all.xls', 'dp_ar_path_all_1.xls', 'dp_ar_path_all_2.xls']

    path_main = path_dir + path_list[2]
    print("path_main", path_main)
    df = pd.read_excel(path_main)
    dp_port_main = df['起飞机场坐标']
    ar_port_main = df['降落机场坐标']
    flight_data_main = df['飞行路径']
    depart_time = []
    arrival_time = []
    path_num = []
    r_main = []
    dp_main = []
    ar_main = []
    dp_ar = []

    simulator_main = flight_simulator(flight_data_main, dp_port_main, ar_port_main)
    route_total_main = []
    for i in range(len(dp_port_main)):
        route = simulator_main.route_total[i]
        dp = simulator_main.depart_port[i]
        ar = simulator_main.arrival_port[i]
        route_new = data_process(route, dp, ar)
        route_total_main.append(route_new)

    path_0 = path_dir + path_list[0]
    df = pd.read_excel(path_0)
    dp_port_0 = df['起飞机场坐标']
    ar_port_0 = df['降落机场坐标']
    flight_data_0 = df['飞行路径']
    simulator_0 = flight_simulator(flight_data_0, dp_port_0, ar_port_0)
    route_total_0 = []
    for i in range(len(dp_port_0)):
        route = simulator_0.route_total[i]
        dp = simulator_0.depart_port[i]
        ar = simulator_0.arrival_port[i]
        route_new = data_process(route, dp, ar)
        route_total_0.append(route_new)

    path_1 = path_dir + path_list[1]
    df = pd.read_excel(path_1)
    dp_port_1 = df['起飞机场坐标']
    ar_port_1 = df['降落机场坐标']
    flight_data_1 = df['飞行路径']
    simulator_1 = flight_simulator(flight_data_1, dp_port_1, ar_port_1)
    route_total_1 = []
    for i in range(len(dp_port_1)):
        route = simulator_1.route_total[i]
        dp = simulator_1.depart_port[i]
        ar = simulator_1.arrival_port[i]
        route_new = data_process(route, dp, ar)
        route_total_1.append(route_new)

    for i in range(len(dp_port_main)):
        #print("第{}对起点_终点处理中".format(i))
        depart = simulator_main.depart_port[i]
        arrival = simulator_main.arrival_port[i]
        route_cur = route_total_main[i]
        for j in range(len(route_total_0[i])):
            a = route_total_0[i]
            if a[j] not in route_cur:
                route_cur.append(a[j])
                #print("route_total_0中找到新路径！")
            else:
                continue

        for k in range(len(route_total_1[i])):
            b = route_total_1[i]
            if b[k] not in route_cur:
                route_cur.append(b[k])
                #print("route_total_1中找到新路径")
            else:
                continue
        """
        # 去掉空路径
        if not route_cur:
            for l in range(len(simulator.depart_port)):
                DP = simulator.depart_port[l]
                AR = simulator.arrival_port[l]
                if depart == DP and arrival == AR and not simulator.route_total[l]:
                    if simulator.route_total[l] not in route_cur:
                        route_cur.append(simulator.route_total[l])
                    else:
                        continue
        """
        # 去掉没有路径的航班
        if len(route_cur) < 1:
            continue


        for l in range(len(simulator.depart_port)):
            DP = simulator.depart_port[l]
            AR = simulator.arrival_port[l]
            if depart == DP and arrival == AR:
                if (DP, AR) in dp_ar:
                    continue
                else:
                    dp_ar.append((depart, arrival))
                    depart_time.append(dp_time[l])
                    arrival_time.append(ar_time[l])


        dp_main.append(dp_port_main[i])
        ar_main.append(ar_port_main[i])

        path_num.append(len(route_cur))

        r_main.append(route_cur)


    data = pd.DataFrame()


    print("############dbg############：", len(depart_time))
    print("#############dbg#########：", len(arrival_time))
    data['起飞机场坐标'] = dp_main
    data['降落机场坐标'] = ar_main
    data['起飞时间'] = depart_time
    data['降落时间'] = arrival_time
    data['可选路径个数'] = path_num
    data['飞行路径'] = r_main


    data.to_excel("find_all_path.xls")


if __name__ == '__main__':
    main()
