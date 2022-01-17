import pandas as pd

def main_1():
    df = pd.read_excel("data_large_scale.xls")
    flight_data = df['飞行路径']
    dp_port = df['起飞机场坐标']
    ar_port = df['降落机场坐标']
    dp_time = df['起飞时间']
    ar_time = df['降落时间']
    path_num = df['可选路径个数']

    flight_data_main = []
    dp_port_main = []
    ar_port_main = []
    dp_time_main = []
    ar_time_main = []
    path_num_main = []

    for i in range(len(dp_port)):
        dpt = int(dp_time[i][-5:-3] + dp_time[i][-2:])
        art = int(ar_time[i][-5:-3] + ar_time[i][-2:])
        # if dpt >= 800 and dpt <= 1200 and art <= 1200:
        # if dpt >= 900 and dpt <= 1200 and art <= 1200:
        # if dpt >= 900 and dpt <= 1200:
        # if dpt >= 1400 and dpt <= 1800 and art <= 1800:
        if dpt >= 1500 and dpt <= 1800 and art <= 1800:
            dp_time_main.append(dp_time[i][-5:-3] + dp_time[i][-2:])
            ar_time_main.append(ar_time[i][-5:-3] + ar_time[i][-2:])
            flight_data_main.append(flight_data[i])
            dp_port_main.append(dp_port[i])
            ar_port_main.append(ar_port[i])
            path_num_main.append(path_num[i])
        else:
            continue

    data = pd.DataFrame()

    data['起飞机场坐标'] = dp_port_main
    data['降落机场坐标'] = ar_port_main
    data['起飞时间'] = dp_time_main
    data['降落时间'] = ar_time_main
    data['可选路径个数'] = path_num_main
    data['飞行路径'] = flight_data_main

    data.to_excel("find_all_path2_3.xls")

# 计算时隙
def interval_compute(time, start = '1500', end = '1800'):
    start = int(start[:-2]) * 60 + int(start[-2:])
    end = int(end[:-2]) * 60 + int(end[-2:])
    time = int(time[:-2]) * 60 + int(time[-2:])
    if time >= start and time <= end:
        return time - start


def main_2():
    df = pd.read_excel("find_all_path2_3.xls")
    flight_data = df['飞行路径']
    dp_port = df['起飞机场坐标']
    ar_port = df['降落机场坐标']
    dp_time = df['起飞时间']
    ar_time = df['降落时间']
    path_num = df['可选路径个数']

    print(dp_time[3])

    for i in range(len(dp_port)):
        dp_time[i] = interval_compute(str(dp_time[i]))
        ar_time[i] = interval_compute(str(ar_time[i]))

    data = pd.DataFrame()

    data['起飞机场坐标'] = dp_port
    data['降落机场坐标'] = ar_port
    data['起飞时间'] = dp_time
    data['降落时间'] = ar_time
    data['可选路径个数'] = path_num
    data['飞行路径'] = flight_data

    # data.to_excel("find_all_path3_large.xls")
    # data.to_excel("find_all_path3.xls")
    data.to_excel("find_all_path3_mid.xls")


if __name__ == '__main__':
    main_1()