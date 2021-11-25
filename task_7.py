from task5 import flight_simulator
import pandas as pd
import numpy as np
import time

df = pd.read_excel("find_all_path3_large.xls")
flight_data = df['飞行路径']
dp_port = df['起飞机场坐标']
ar_port = df['降落机场坐标']

def conflict_relation_of_flight(id1, id2):
    """
    :param i: i号航班
    :param j: j号航班
    :return: 是否冲突 True --- 冲突 False --- 不冲突
    """
    # print("id1, id2", id1, id2)
    flight_num = 2
    dp_time = []
    for i in range(flight_num):
        dp_time.append(0)
    path_select = []
    for i in range(flight_num):
        path_select.append(0)

    simulator = flight_simulator(dp_time, path_select, specific_id=[id1, id2])
    simulator.flight_update()
    # print("冲突数量:", simulator.conflict_num)
    if simulator.conflict_num == 0:
        return False
    else:
        return True


def relation_init():
    relation = []
    for i in range(len(dp_port)):
        print(i)
        s = []
        for j in range(len(dp_port)):
            if i == j:
                continue
            else:
                if conflict_relation_of_flight(i, j):
                    s.append(j)
        relation.append(s)

    file = open('relation_large.txt', 'w')
    for fp in relation:
        file.write(str(fp))
        file.write('\n')
    file.close()

def dataprocess():
    file = open('relation_large.txt', 'r')
    list_read = file.readlines()
    relation = []
    for a in list_read:
        a = a.replace('[', '')
        a = a.replace(']', '')
        a = a.replace(',', '')
        a = a.split()
        for i in range(len(a)):
            a[i] = int(a[i])
        relation.append(a)

    return relation


def graph_init():
    """
    graph = [
    {'point': 0, 'edge': [1,2,3], 'visit': 0},
    {'point': 1, 'edge': [0,3], 'visit': 0},
    {'point': 2, 'edge': [0], 'visit': 0},
    {'point': 3, 'edge': [0,1], 'visit': 0},
    {'point': 4, 'edge': [5], 'visit': 0},
    {'point': 5, 'edge': [4,6], 'visit': 0},
    {'point': 6, 'edge': [5], 'visit': 0}
    ]
    :return: graph
    """
    relation = dataprocess()
    graph = []
    for i in range(len(dp_port)):
        dict = {'point': i, 'edge': relation[i], 'visit': 0}
        graph.append(dict)

    return graph


graph = graph_init()
print("graph", graph)


def DFS(v, path = []):
    path.append(v['point'])
    v['visit'] = 1
    # print("path:", path)
    for i in v['edge']:
        if graph[i]['visit'] == 0:
            DFS(graph[i], path)

    return path

def flight_divide():
    blk = 0
    path_all = []
    for i in range(len(graph)):
        v = graph[i]
        if v['visit'] == 0:
            path = DFS(v, path=[])
            path_all.append(path)
            blk += 1
        else:
            continue

    return path_all, blk

if __name__ == '__main__':
    start = time.time()
    divide, num = flight_divide()
    # relation_init()
    end = time.time()
    print("花费时间:", end - start)
    print("divide:", divide)
    print("分类个数:", num)