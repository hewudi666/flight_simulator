from task5 import flight_simulator
import numpy as np
import time

v = np.load('E:\\飞行数据\\20190901\\下午\\大规模\\default_v_1539.npy')
print("v", v.shape)
path_len = np.load('E:\\飞行数据\\20190901\\下午\\大规模\\path_length_1539.npy')
print("path_len_shape", path_len.shape)
print("path_len:", path_len)
# path_select = np.zeros(810).astype(np.int32)
# dp_time = np.zeros(810).astype(np.int32)
# start = time.time()
# simulator = flight_simulator(dp_time, path_select)
# simulator.velocity_input(v)
# simulator.flight_update()
# end = time.time()
# # 更新时间
# print("更新花费时间：", end - start)
# # 冲突次数
# print("冲突次数：", simulator.conflict_num)
# # 飞行器到达状态：0表示未到达, 1表示已到达
# print("到达状态：", simulator.teriminal)
# # 延误时间
# print("延迟时间：", simulator.delay_time)
# # 总延迟时间
# print("总延迟时间:", sum(simulator.delay_time))