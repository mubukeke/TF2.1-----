import tensorflow as tf

w = tf.Variable(4)
w.assign(1)   # w -= 1  w = w - 1
print(w)
