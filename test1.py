# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import matplotlib.colors as mcolors
# from task5 import flight_simulator
#
# colors = list(mcolors.TABLEAU_COLORS.keys())
#
# flight_num = 332
# dp_time = []
# for i in range(flight_num):
#     dp_time.append(0)
# path_select = []
# for i in range(flight_num):
#     path_select.append(0)
# simulator = flight_simulator(dp_time, path_select)
# simulator.flight_update()
# conflict = simulator.conflict_point
#
# def data_read():
#     flight_num = 332
#     file = open('flight_route_data.txt', 'r')
#     list_read = file.readlines()
#     flight_route_total = []
#     for a in list_read:
#         flight_route = []
#         a = a.replace('[', '')
#         a = a.replace(']', '')
#         a = a.replace('(', '')
#         a = a.replace(')', '')
#         a = a.replace(',', '')
#         a = a.split()
#         for i in range(flight_num):
#             lon = float(a[i * 2])
#             lat = float(a[i * 2 + 1])
#             point = (lon, lat)
#             flight_route.append(point)
#         flight_route_total.append(flight_route)
#
#     return flight_route_total
#
# route_total = data_read()
# route_total.append(conflict)
#
# fig, ax = plt.subplots()
# flight_num = len(route_total[0])
# num_total = len(route_total[0]) + 1
# time_num = len(route_total)
# ln = [[]] * num_total
#
# for i in range(flight_num):
#     ln[i], = ax.plot([], [], color=mcolors.TABLEAU_COLORS[colors[i % 10]], marker='.')
# ln[flight_num], = ax.plot([], [], 'ro')
#
#
# plt.grid(True)
# plt.axis("equal")
# plt.gca().set_xbound([70, 130])
# plt.gca().set_ybound([15, 52])
# x = []
# y = []
# cx = []
# cy = []
#
# for i in range(flight_num):
#     rx = []
#     ry = []
#     for j in range(time_num):
#         p = route_total[j][i]
#         rx.append(p[0])
#         ry.append(p[1])
#     x.append(rx)
#     y.append(ry)
#
# for k in range(len(conflict)):
#     q = conflict[k]
#     cx.append(q[0])
#     cy.append(q[1])
#
# print("x",x)
# print("cx",cx)
# def init():
#     ax.set_xlim(70, 130)
#     ax.set_ylim(15, 52)
#     return ln,
#
#
# def update(i):
#     if len(route_total[i]) == flight_num:
#         for j in range(flight_num):
#             ln[j].set_data(x[j][0:i], y[j][0:i])
#     else:
#         x[-2].extend(cx)
#         y[-2].extend(cy)
#         ln[-1].set_data(x[-2], y[-2])
#     return *ln,
#
#
# anim = animation.FuncAnimation(fig, update, frames=np.arange(0, time_num), interval=10, blit=True)
#
#
# # anim.save('test_animation.mp4', fps=30)
# # anim.save('test_animation1.gif', writer='imagemagic', fps=30)
# plt.show()
#
#
#
