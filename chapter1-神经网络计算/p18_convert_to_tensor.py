import tensorflow as tf
import numpy as np

a = np.arange(0, 8, 2)  # [0, 8) 间隔为2
b = tf.convert_to_tensor(a, dtype=tf.int64)
print(a)
print(b)
