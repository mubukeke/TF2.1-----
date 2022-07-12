import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

Y_c = [ ['red' if y else 'blue'] for y in y_train ]

# 数据类型转换
x_train = tf.cast(x_train, dtype=tf.float32)
y_train = tf.cast(y_train, dtype=tf.float32)

# from_tensor_slices 生成训练集，再划分batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
# for batch in train_db:
    # print("batch:", batch)

# 搭建神经网络结构 输入层两个特征（2个神经元），一个隐藏层设置为11个神经元，一个输出神经元
# tf.Variable() 保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

# 设置超参数，构建训练
lr = 0.01
epoch = 400

for epoch in range(epoch):
    for batch, (x_train_batch, y_train_batch) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # 第一个隐含层计算
            hidden_layer_1 = tf.matmul(x_train_batch, w1) + b1  # 计算乘加
            hidden_layer_1 = tf.nn.relu(hidden_layer_1)   # relu激活
            # 输出层计算
            y = tf.matmul(hidden_layer_1, w2) + b2  # 计算乘加，输出层不用激活

            # 采用MSE作为损失函数 mse = mean(sum(y_ - y)^2)
            loss_mse = tf.reduce_mean(tf.square(y - y_train_batch))

        # 对各个参数求梯度
        grads = tape.gradient(loss_mse, [w1, b1, w2, b2])
        # 更新参数
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    if epoch % 20 == 0:
        print("epoch: ", epoch, "loss: ", float(loss_mse))

# 预测部分
print("***********predict*************")

x_axis, y_axis = np.mgrid[-3:3:0.1, -3:3:0.1]
x_axis_flatten = x_axis.ravel()
y_axis_flatten = y_axis.ravel()
grid = np.c_[x_axis_flatten, y_axis_flatten]
grid = tf.cast(grid, tf.float32)

probs = []
for cross_dot in grid:
    h1 = tf.matmul([cross_dot], w1) + b1
    h1 = tf.nn.relu(h1)
    cross_dot_pred = tf.matmul(h1, w2) + b2
    probs.append(cross_dot_pred)

# 取第0列给x1, 取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]

probs = np.array(probs).reshape(x_axis.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
plt.contour(x_axis, y_axis, probs, levels=[0.5])
plt.show()
