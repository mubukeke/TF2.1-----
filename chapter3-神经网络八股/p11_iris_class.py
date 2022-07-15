# 导入相关的模块和类
from p11_IrisModel import IrisModel
import tensorflow as tf
import numpy as np
from sklearn import datasets

# 数据-训练，测试
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 使用class来构建网络结构。更通用-顺序网络/跳连网络 皆可方便构建
model = IrisModel()
# model.print()

# compile 配置网络参数-优化器，损失函数，计算准确率方式
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

# fit 设置参数训练-训练特征，标签，batch，epoch，测试集，多少轮次使用一次测试集。
model.fit(x_train, y_train,
          batch_size=32, epochs=500,
          validation_split=0.3,
          validation_freq=20)

# summary 输出网络结构和参数信息
model.summary()


