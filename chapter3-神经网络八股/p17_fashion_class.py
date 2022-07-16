# import 相关模块
import tensorflow as tf
import numpy as np
from p17_FashionModel import FashionModel

# 数据集（训练集，测试集）
fashion = tf.keras.datasets.fashion_mnist   # 得到fashion数据集的模块
(x_train, y_train), (x_test, y_test) = fashion.load_data()

# 数据归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 使用class方式来构建神经网络结构
model = FashionModel()

# compile 配置网络 （优化器，损失函数，计算损失函数方法）
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # y_pred 是概率
              metrics=["sparse_categorical_crossentropy"])

# fit 设置网络训练参数-训练特征集合，标签，batch，epoch，测试集/测试集占数据集比例，多少epoch使用测试集验证一次精度
model.fit(x_train, y_train,
          batch_size=32, epochs=10,
          validation_data=(x_test, y_test),
          validation_freq=1)

# summary 打印网络结构，参数信息
model.summary()
