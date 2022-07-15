import tensorflow as tf
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 括号是元组tuple，表示不可被修改

# 可视化最后一个测试图片
plt.imshow(x_train[59999], cmap='gray')
plt.show()

# 打印训练集的第一个 特征和标签
print("训练集第一个特征，x_train[0]:\n", x_train[0])      # 28x28 的二维矩阵，每个元素就是该位置像素点对应的灰度值
print("训练集第一个特征对应标签，y_train[0]:\n", y_train[0])   # 该手写数字对应的标签

# 打印整个训练集合和测试集合的特征与标签形状
print("训练集特征的形状，x_train.shape = ", x_train.shape)
print("训练集特征对应标签的形状，y_train.shape = ", y_train.shape)
print("测试集特征的形状，x_test.shape = ", x_test.shape)
print("测试集特征对应标签的形状，y_test.shape = ", y_test.shape)

print("测试集特征，x_test\n", x_test)  # 不是tensor张量，而是直接的numpy三维数组，里面存放10000个二维灰度值矩阵
print("测试集特征对应标签，y_test\n", y_test)   # 是numpy一维数组，里面存放10000个标签（0-9）
