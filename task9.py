import pandas as pd
import numpy as np
from task5 import flight_simulator
import time

"""
机场数，waypoints，总航线数量
"""

flight_num = 925
dp_time = []
for i in range(flight_num):
    dp_time.append(0)
path_select = []
for i in range(flight_num):
    path_select.append(0)
simulator = flight_simulator(dp_time, path_select)
dp_port = simulator.depart_port
ar_port = simulator.arrival_port
dp_port.extend(ar_port)
all_port_num = len(list(set(dp_port)))
flight_path_num = sum(simulator.path_num)
print("航班数量", flight_num)
print("机场数量", all_port_num)
waypoint = []
for i in range(flight_num):
    route = simulator.route_total[i]
    for j in range(simulator.path_num[i]):
        waypoint.extend(route[j])
print("航路点数量", len(list(set(waypoint))))
print("总航线数量", flight_path_num)


