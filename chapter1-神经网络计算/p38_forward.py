import tensorflow as tf

x = tf.constant([[5.8, 4.0, 1.2, 0.2]])
w = tf.random.truncated_normal([4, 3], mean=0, stddev=1)
b = tf.random.truncated_normal([1, 3], mean=0, stddev=1)

print("input data: ", x)
print("random init weights: ", w)
print("random init bias: ", b)

y = tf.matmul(x, w) + b
print("forward output: ", y)

x = tf.constant([[5.8, 4.0, 1.2, 0.2]])
w = tf.constant([[-0.8, -0.34, -1.4],
                 [0.6, 1.3, 0.25],
                 [0.5, 1.45, 0.9],
                 [0.65, 0.7, -1.2]])
b = tf.constant([[2.52, -3.1, 5.62]])
y = tf.matmul(x, w) + b

print("x: ", x)
print("w: ", w)
print("b: ", b)
print("y: ", y)

soft_max = tf.nn.softmax(y)
soft_max_dim = tf.squeeze(soft_max)
print("sotfmax output: ", soft_max)
print("sotfmax dim: ", soft_max_dim)

y_dim = tf.squeeze(y)  # tf.squeeze(y) 就是把[[1, 2, 3]] -> [1, 2, 3] 去掉多余的维度。
y_propagation = tf.nn.softmax(y_dim)
print("y_dim: ", y_dim)
print("y_propagation: ", y_propagation)
