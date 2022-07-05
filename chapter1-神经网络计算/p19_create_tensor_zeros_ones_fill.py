import tensorflow as tf

a = tf.zeros([2, 3], dtype=tf.int64)
b = tf.ones(4, dtype=tf.float64)
c = tf.fill([2, 3, 4], 1, name="ones_matric")
print(a)
print(b)
print(c)
