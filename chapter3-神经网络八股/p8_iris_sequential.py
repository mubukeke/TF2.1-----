# 神经网络搭建八股

# 第一股：import 相关模块
import tensorflow as tf
from sklearn import datasets
import numpy as np

# 第二股：划分训练，测试集（特征 + 标签）
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(10)
np.random.shuffle(x_train)
np.random.seed(10)
np.random.shuffle(y_train)
tf.random.set_seed(10)

x_train = x_train[:-30]
y_train = y_train[:-30]
x_test = x_train[-30:]
y_test = y_train[-30:]

# 第三股：Sequential 搭建网络结构
# 使用 TF 中的 keras 库中的 Sequential 结构搭建网络结构（顺序型）
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())
])

# 第四股：Compile 配置神经网络训练方法
# 配置优化器，损失函数，等等
model.compile(optimizer="sgd",
              # loss="sparse_categorical_crossentropy",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 上面的神经网络全连接层输出时使用了softmax，输出为概率，所以为False
              metrics=["sparse_categorical_accuracy"])

# 第五股：fit 执行训练过程
model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=500,
          # validation_split=0.3,
          validation_data=[x_test, y_test],   # [], () 都行
          validation_freq=20)

# 第六股：summary 打印网络结构和训练参数
model.summary()
