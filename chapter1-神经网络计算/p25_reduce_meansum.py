import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [2., 2., 3.]])  # 都是整数，默认都是int32, 有小数，默认float32
print(x)
print(tf.reduce_mean(x))  # 整数的除法，和C++一致，自动取整的
print(tf.reduce_sum(x, axis=0))
print(tf.reduce_sum(x, axis=1))
print(tf.reduce_sum(x))

