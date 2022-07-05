import tensorflow as tf

d = tf.random.normal([3, 3], mean=0, stddev=1)
print(d)

e = tf.random.truncated_normal([3, 3], mean=0, stddev=1)
print(e)
