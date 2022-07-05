import tensorflow as tf

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant([3.0, 4.0, 5.0]))
    loss = tf.square(w)
grad = tape.gradient(loss, w)
print(grad)

