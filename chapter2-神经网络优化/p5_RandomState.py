
import tensorflow as tf
import numpy as np

rdm = np.random.RandomState(seed=1)  # [0, 1)自定义维度的伪随机数
a = rdm.rand()   # 生成一个随机标量
b = rdm.rand(3, 2)   # [3, 2]

print("a:", a)
print("b:", b)
