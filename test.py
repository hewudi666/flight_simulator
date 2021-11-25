from task5 import flight_simulator
import numpy as np
import time

v = np.load("default_v_1435.npy")
print("v", v)
print(v.shape)

# flight_num = 332
# dp_time = []
# for i in range(flight_num):
#     dp_time.append(0)
# path_select = []
# for i in range(flight_num):
#     path_select.append(0)
#
# simulator = flight_simulator(dp_time, path_select)
# waypoint = []
# for i in range(flight_num):
#     route = simulator.flight_path_total[i]
#     for j in range(1, len(route) - 1):
#         p = route[j]
#         if p in waypoint:
#             continue
#         else:
#             waypoint.append(p)
#
#
# print("num", len(waypoint))


# dp_time = np.load('dp_time.npy')
# path_select = np.load('path_select.npy')
# group_id = list(np.load('group_idx.npy'))
# # group_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 101, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200, 201, 202, 203, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 316, 318, 319, 320, 321, 323, 324, 325, 326, 327, 329, 330, 331]
#
# # false_id = [20, 33, 38, 43, 44, 45, 53, 60, 61, 68, 69, 75, 80, 83, 87, 101, 103, 104, 105, 110, 111, 112, 116, 123, 124, 126, 134, 151, 156, 164, 166, 171, 179, 180, 182, 183, 185, 199, 204, 208, 214, 225, 242, 254, 264, 267, 270, 283, 287]
# # a = path_select[false_id]
# # print("超出索引path_select", a)
#
# start = time.time()
# simulator = flight_simulator(dp_time, path_select, specific_id=group_id)
# end = time.time()
# print("初始化花费时间:", end - start)
# simulator.flight_update()
# end1 = time.time()
# # 更新时间
# print("更新花费时间：", end1 - end)
# # 冲突次数
# print("冲突次数：", simulator.conflict_num)
# # 飞行器到达状态：0表示未到达, 1表示已到达
# print("到达状态：", simulator.teriminal)
# # 延误时间
# print("延迟时间：", simulator.delay_time)
# # 总延迟时间
# print("总延迟时间:", sum(simulator.delay_time))
#
# simulator.plot()

#graph = [
#     {'point': 0, 'edge': [1,2,3], 'visit': 0},
#     {'point': 1, 'edge': [0,3], 'visit': 0},
#     {'point': 2, 'edge': [0], 'visit': 0},
#     {'point': 3, 'edge': [0,1], 'visit': 0},
#     {'point': 4, 'edge': [5], 'visit': 0},
#     {'point': 5, 'edge': [4,6], 'visit': 0},
#     {'point': 6, 'edge': [5], 'visit': 0}
# ]
#
# def DFS(v, path = []):
#     path.append(v['point'])
#     v['visit'] = 1
#     # print("path:", path)
#     for i in v['edge']:
#         if graph[i]['visit'] == 0:
#             DFS(graph[i], path)
#
#     return path
#
#
# blk = 0
# path_all = []
# for i in range(len(graph)):
#     v = graph[i]
#     if v['visit'] == 0:
#         path = DFS(v, path = [])
#         path_all.append(path)
#         blk += 1
#     else:
#         continue
#
# print(blk)
# print(path_all)











