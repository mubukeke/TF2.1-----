import tensorflow as tf
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([11, 22, 33, 44])
c_squared_blanket = np.vstack([a, b])  # 用[]存储合并后的array
c_thephesie = np.vstack((a, b))  # 用()存储合并后的array
print("c_[]: ", c_squared_blanket)
print("c_(): ", c_thephesie)
