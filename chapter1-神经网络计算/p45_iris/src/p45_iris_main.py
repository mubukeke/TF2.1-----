import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为特征和标签
x_features = datasets.load_iris().data
y_labels = datasets.load_iris().target
y_classes = datasets.load_iris().target_names
# print(datasets.load_iris())
# print(y_classes)

# 随机打乱数据 设置种子，故每次生成的伪随机数一样，保证代码可重复性与易分享性
np.random.seed(0)
np.random.shuffle(x_features)
np.random.seed(0)
np.random.shuffle(y_labels)
tf.random.set_seed(0)

# 乱序数据集划分训练（前120组）、测试（后30组）。保证训练数据和测试数据不见面
x_train = x_features[:-30]
y_train = y_labels[:-30]
x_test = x_features[-30:]
y_test = y_labels[-30:]
# print(x_test)

# 类型转换一致
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

# 将特征与标签配对，并划分成batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# 使用Tensorflow搭建网络，四个神经元输入，三个神经元输出的全连接层。只需设置这一层的参数即可
# tf.Variable() 表示参数可训练
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

# 设置训练所需超参数
classes = 3
lr = 0.1
epoch = 500
loss_all = 0   # 每个epoch中有四个batch，每个batch会算一个loss,loss_all是计算平均
train_loss_results = []   # 没轮loss记录，用于绘制loss曲线
test_acc = []   # 每轮acc记录，用于绘制acc曲线

# 训练部分
for epoch in range(epoch):
    for batch, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_hat = tf.one_hot(y_train, depth=classes)
            loss = tf.reduce_mean(tf.square(y - y_hat))   # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()
        # 每个batch后，loss对参数求导
        # 这里只有一层网络，损失函数直接就是 [w1, b1] 的函数，所以链式求导法则在这里就是求一次导，即可
        gradients = tape.gradient(loss, [w1, b1])

        # 参数自更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        # 每一个batch，w1,b1自更新一次
        w1.assign_sub(lr*gradients[0])
        b1.assign_sub(lr*gradients[1])
    # print("batch is ", batch) # batch从 0,1,2,3 所以最后这里为3，一共3+1=4个batch
    print("Epoch {}, loss: {}".format(epoch, loss_all/(batch+1)))
    train_loss_results.append(loss_all/(batch+1))
    loss_all = 0   # loss_all归零，重新记录下一个epoch的所有batch的loss和

    # 测试集部分
    total_correct, total_number = 0, 0
    for batch, (x_test, y_test) in enumerate(test_db):
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        prediction = tf.argmax(y, axis=1)   # 返回最大值的索引，即预测的分类
        prediction = tf.cast(prediction, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(prediction, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test acc: ", acc)
    print("----------------------------------")

# 绘制 loss 曲线
plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results, label="$MSE loss$")
plt.legend()
plt.show()

# 绘制 Accuracy 曲线
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()


# inference 使用训练好的参数做前向传播，自己设计一个1杂色鸢尾。
# 推理结果是2，弗吉尼亚鸢尾。可能数据造的不对，但是明白了前向推理过程
print("w1: ", w1)
print("b1: ", b1)
# 输入自己设计的新的数据（花萼长>花萼宽 and 花瓣长/花瓣宽 > 2 则 1杂色鸢尾）
x_new = tf.constant([[10, 3, 7.4, 2.1]], dtype=w1.dtype)
# 使用tf.matmul注意，x是tf.constant([1,2,3,4]) 一维向量；w1是tf.Variable([4,3])二维矩阵
# 所以需要注意 x=tf.constant([[1,2,3,4]]) 设置为二维的
y_new_hat = tf.matmul(x_new, w1) + b1
print("y_new_hat:\n", y_new_hat)
y_new_hat = tf.squeeze(y_new_hat)
y_new_hat = tf.nn.softmax(y_new_hat)
print("y_new_hat_softmax:\n", y_new_hat)
# tf.reduce_max 与　tf.argmax
# 返回向量一行的最大值   返回向量一行最大值的索引
pred_index = tf.argmax(y_new_hat)
print("output class is:\n", pred_index)
