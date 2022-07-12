"""   2022-7-11
在老师原来题目：生成随机数[x1, x2]，设定y_ = x1 + x2 + 噪声. 使用y = KX 进行拟合。epoch15000次后，W_1~1,W_2~1
修改：设定y_= 2*x1 + 3*x2 + 10 + 噪声。 使用 y=KX+B进行拟合
"""
import tensorflow as tf
import numpy as np

# 制造数据集
SEED = 10
rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2)
y_ = [ [2*x1 + 3*x2 + 10 + (rdm.rand() / 10 - 0.05)] for (x1, x2) in x]  # 构造伪gt，y_=2*x1+3*x2+10+噪声
x = tf.cast(x, dtype=tf.float32)

# 搭建网络 （只有一个输出层，即MP模型）
"""
输入特征  o   
              o  输出预测结果      训练求解参数：w1 = [w11, w21].transpose,  b   
输入特征  o
"""
w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))
b1 = tf.Variable(tf.random.normal([1], stddev=1, seed=1))

# 参数优化（训练）
epoch = 20000
lr = 0.02   # 学习率不变可以学习得到正确参数，使用指数学习率衰减，反而学的不准确了
# LR_BASE = 0.03
# LR_DECAY = 0.8  # 学习率衰减率
# LR_STEP = 2  # 喂入多少轮后，更新一次学习率

for epoch in range(epoch):
    # lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1) + b1
        loss_mse = tf.reduce_mean(tf.square(y - y_))   # 均方误差 mean square

    grads = tape.gradient(loss_mse, [w1, b1])
    w1.assign_sub(lr*grads[0])
    b1.assign_sub(lr*grads[1])

    if epoch % 50 == 0:
        print("After {} training epoches, w1 is {}\nb1 is {}\n".format(epoch, w1.numpy(), b1.numpy()))

print("Final w1 is: ", w1.numpy())
print("Final b1 is: ", b1.numpy())
