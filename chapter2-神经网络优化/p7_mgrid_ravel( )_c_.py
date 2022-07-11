import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# x: 1,2  y:2,2.5,3,3.5,4
x, y = np.mgrid[1:3:1, 2:4:5j]
x_flatton = x.ravel()
y_flatton = y.ravel()

grid = np.c_[x_flatton, y_flatton]

print("x:", x)
print("y:", y)
print("grid:", grid)
print("x_flatton:", x_flatton)
print("y_flatton:", y_flatton)

plt.title("np.mgrid generate 2D dots")
plt.xlabel("x: 1, 2 two ")
plt.ylabel("y: 2, 2.5, 3, 3.5, 4 five")
# plt.plot(grid[0][0], grid[0][1], marker='*', color='r', linestyle='none')
# plt.plot(grid[1][0], grid[1][1], marker='*', color='r', linestyle='none')

# 一组点绘制
x_dots = grid[:, 0]   # 效果等于 x_flatton
y_dots = grid[:, 1]   # 效果等于 y_flatton
# print("x_dots:", x_dots)
# print("y_dots:", y_dots)
plt.scatter(x_dots, y_dots, marker='*', color='b')
plt.show()

# 单点绘制
for index, dot in enumerate(grid):
    # print("dot:", dot)
    print("%i: dot[%f][%f]" % (index, dot[0], dot[1]))
    plt.plot(dot[0], dot[1], marker='*', color='r', linestyle='none')
plt.show()
