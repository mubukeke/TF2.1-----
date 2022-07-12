import tensorflow as tf
import numpy as np

SEED = 10
COST = 99
PROFIT = 1

rdm = np.random.RandomState(SEED)
x = rdm.rand(32, 2)
y_ = [ [x1 + x2 + (rdm.rand() / 10 - 0.05)] for (x1, x2) in x ]
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 10000
lr = 0.002
"""
预测商品销量，预测多了，损失的是成本; 预测少了，损失的是利润
利润 ！= 成本， 所以需要自定义损失函数来使得利益最大化。
f(y_, y)  =  PROFIT * (y_ - y); y < y_  预测的少了，损失的是利润（PROFIT加权）
          =  COST * (y - y_);  y >= y_  预测的多了，损失的是成本（COST加权）
"""
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ - y)*PROFIT))

    grads = tape.gradient(loss, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print("After %d training epoch,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())