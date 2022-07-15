# 导入相关的模块
import tensorflow as tf
from p15_MnistModel import MnistModel

# 数据集-训练集-验证集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 数据归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 搭建网络结构 - class方式
model = MnistModel()

# compile 配置网路参数-优化器，损失函数，精度计算
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

# fit 训练参数设置
model.fit(x_train, y_train,
          batch_size=32, epochs=10,
          validation_data=(x_test, y_test),
          validation_freq=1)

# summary 打印网络结构，参数信息
model.summary()
