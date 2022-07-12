import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import  time

# 导入数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱
np.random.seed(10)
np.random.shuffle(x_data)
np.random.seed(10)
np.random.shuffle(y_data)
tf.random.set_seed(10)

# 打乱后数据划分训练和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 保持数据类型一致
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

# from_tensor_slices 函数变成 特征标签对，并划分batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 构建网络 只有输出层的一层全连接神经网络
w1 = tf.Variable(tf.random.normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.normal([3], stddev=0.1, seed=1))

# 设置超参数
# lr = 0.1
lr = 0.01
train_loss_results = []
test_acc = []
# epoch = 500
epoch = 100
loss_all = 0
classes = 3

# 训练部分
now_time = time.time()
for epoch in range(epoch):
    for batch, (x_train_batch, y_train_batch) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train_batch, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train_batch, depth=classes)
            loss = tf.reduce_mean(tf.square(y - y_))
            loss_all += loss.numpy()

        # loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 参数自更新 SGD 优化
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    # 每个epoch，打印loss
    print("Epoch {}, loss {}".format(epoch, loss_all/(batch+1)))
    train_loss_results.append(loss_all/(batch+1))
    loss_all = 0

    # 测试部分
    total_correct, total_number = 0, 0
    for batch_test, (x_test_batch, y_test_batch) in enumerate(test_db):
        y = tf.matmul(x_test_batch, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test_batch.dtype)

        correct = tf.cast(tf.equal(pred, y_test_batch), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test_batch.shape[0]

    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc: ", acc)
    print("--------------------")

total_time = time.time() - now_time
print("total_time ", total_time)

plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.xlabel("Loss")
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.xlabel("Accuracy")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()