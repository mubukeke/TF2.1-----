# import 相关的模块
import tensorflow as tf

# 加载训练集/测试集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 数据做归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# sequential 搭建顺序网络结构
# 首先将二维像素矩阵拉直，当作特征输入；一个128节点的全连接层+relu激活；一个10节点的输出层+softmax
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# cmopile 进行网络参数配置-数据集，优化方法，损失函数，正则化，定义如何计算y_pred与y_gt(两个都值？一个概率，一个独热码？等)
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 输出层进过了softmax,变成了概率，所以这里参数维False
              metrics=["sparse_categorical_accuracy"])   # y_gd 是数值，y_pred 是进过了softmax的概率值，所以选aparse_catagorical_accuracy

# fit 模型训练-设置batch，epoch，测试集占数据集的百分比，多少个epoch后进行一次测试集来进行前向推理计算进度
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)

# summary 网络结构和参数信息 - 输出网络层数，可训练的参数规模
model.summary()

