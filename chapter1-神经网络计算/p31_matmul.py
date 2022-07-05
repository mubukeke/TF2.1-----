import tensorflow as tf

a = tf.ones([3, 2])
print(a)
b = tf.fill([2, 3], 3.)
print(b)

print(tf.matmul(a, b))  # 矩阵乘是使用 matmul，tf.multiply 是对位点乘
