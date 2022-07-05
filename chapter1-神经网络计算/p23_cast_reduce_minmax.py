import tensorflow as tf

x1 = tf.constant([6., 2., 3.], dtype=tf.float64)  # type is float 需要加点
print(x1)

x2 = tf.cast(x1, tf.float32)
print(x2)

print("min value: ", tf.reduce_min(x2))
print("max value: ", tf.reduce_max(x2))
