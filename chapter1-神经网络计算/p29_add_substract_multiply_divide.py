import tensorflow as tf

a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)

print(a)
print(b)

print(tf.add(a, b))
print(tf.subtract(a, b))
print(tf.multiply(a, b))  # 对应位置相乘，还是同维度。
print(tf.divide(b, a))

print(tf.pow(b, 3))
print(tf.square(b))
print(tf.sqrt(b))


