"""
任务目标
提取航班号等6个信息，保存为dataframe格式

1.根据飞行数据提取航班号，起飞降落机场代号，起飞降落时间，飞行航路

2.根据起飞降落机场代号在“经纬度224个机场”中提取经纬度坐标(如果搜索不到机场，则剔除)

3.根据飞行航路提取航路点，再根据"waypoints"中的searchName提取经纬度，并进行单位转换
(其中搜索不到的航路点剔除跳过)，最后一头一尾加上起点终点和经纬度信息

4.整理表格数据并保存
"""

import pandas as pd


def dataprocess(filename):

    df = pd.read_excel(str(filename) + ".xls")
    airport = pd.read_excel("经纬度224个机场.xls")
    waypoint = pd.read_excel("waypoint.xlsx")

    #提取航班号，起飞降落机场代号，起飞降落时间，飞行航路
    columns = [' FLIGHTID', ' P_DEPAP',' P_ARRAP',' P_DEPTIME',' P_ARRTIME',' P_ROUTE']
    data_flight = pd.DataFrame(data=df,columns=columns)
    #print(data_flight)
    airport = pd.DataFrame(airport)

    #飞行时间数据类型转换并处理
    data_flight['起飞时间'] = data_flight[' P_DEPTIME'].astype(str)
    data_flight['降落时间'] = data_flight[' P_ARRTIME'].astype(str)

    #经纬度转换
    def transform(du, fen, miao):
        result = float(du) + float(fen) / 60 + float(miao) / 3600
        return float(result)


    depart_pos = []
    arrival_pos = []
    #根据起飞降落机场代号查找对应经纬度信息
    for index, row in data_flight.iterrows():
        depart_id = row[' P_DEPAP']
        arrival_id = row[' P_ARRAP']
        index1 = airport.loc[airport['英文代码'] == depart_id]
        index2 = airport.loc[airport['英文代码'] == arrival_id]
        if index1.empty or index2.empty:
            data_flight = data_flight.drop(index, axis=0)
            continue
        depart = (round(index1['经度'].values[0], 2),round(index1['纬度'].values[0], 2))
        arrival = (round(index2['经度'].values[0], 2),round(index2['纬度'].values[0], 2))
        #print(type(depart))
        depart_pos.append(depart)
        arrival_pos.append(arrival)
        data_flight.loc[index, '起飞时间'] = data_flight.loc[index, '起飞时间'][0:4] + '.' + data_flight.loc[index, '起飞时间'][4:6] + '.' + data_flight.loc[index, '起飞时间'][6:8] + ',' + \
                              data_flight.loc[index, '起飞时间'][8:10] + ':' + data_flight.loc[index, '起飞时间'][10:12]
        data_flight.loc[index, '降落时间'] = data_flight.loc[index, '降落时间'][0:4] + '.' + data_flight.loc[index, '降落时间'][4:6] + '.' + data_flight.loc[index, '降落时间'][6:8] + ',' + \
                              data_flight.loc[index, '降落时间'][8:10] + ':' + data_flight.loc[index, '降落时间'][10:12]

    data_flight['起飞机场坐标'] = depart_pos
    data_flight['降落机场坐标'] = arrival_pos
    #删除多余列
    data_flight = data_flight.drop(' P_DEPAP', axis=1)
    data_flight = data_flight.drop(' P_ARRAP', axis=1)
    data_flight = data_flight.drop(' P_DEPTIME', axis=1)
    data_flight = data_flight.drop(' P_ARRTIME', axis=1)


    #根据waypoint代号查找对应经纬度信息
    waypoint = pd.DataFrame(waypoint)
    #飞行路径合集
    flight_route_total = []
    for Index, Row in data_flight.iterrows():
        route = data_flight.loc[Index, ' P_ROUTE']
        if isinstance(route,float):
            data_flight = data_flight.drop(Index, axis=0)
            continue
        print(route)
        #去掉航路段
        route_new = route.split()[::2]
        flight_route = [data_flight.loc[Index, '起飞机场坐标']]
        for i in range(len(route_new)):
            id = waypoint.loc[waypoint['searchName'] == route_new[i]]
            if id.empty:
                continue

            lon = id['longitude'].values[0].astype(str)
            lat = id['latitude'].values[0].astype(str)

            lon_trans = transform(lon[0:-4], lon[-4:-2], lon[-2:])
            lat_trans = transform(lat[0:-4], lat[-4:-2], lat[-2:])
            #print("lon_trans：", type(lon_trans))

            point = (round(lon_trans, 2), round(lat_trans, 2))

            flight_route.append(point)
        flight_route.append(data_flight.loc[Index, '降落机场坐标'])
        flight_route_total.append(flight_route)


    data_flight['飞行路径'] = flight_route_total
    data_flight = data_flight.drop(' P_ROUTE', axis=1)

    #处理后数据
    print(data_flight)

    save_path = "C:\\Users\\lenovo\\PycharmProjects\\project\\处理后飞行数据\\" + str(filename[-8:])

    # data_flight.to_excel(save_path + "飞行数据_处理后.xls")
    #data_flight.to_csv(save_path + "飞行数据_处理后.csv")


def main():

    for i in range(1, 2):
        filename = "E:\\飞行数据\\2019年数据\\" + "2019100" + str(i)
        dataprocess(filename)

if __name__ == '__main__':
    main()







