# 导入相关的模块 module
import tensorflow as tf

# 训练集,测试集
fashion = tf.keras.datasets.fashion_mnist   # 加载数据集模块
(x_train, y_train), (x_test, y_test) = fashion.load_data()
# 数据归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential 搭建顺序网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2()),
    tf.keras.layers.Dense(10, activation="softmax")
])

# compile 配置模型参数(优化器,损失函数)
model.compile(optimizer="adam",
              # loss="sparse_categorical_crossentropy",   # 默认也是 Fasle(输出经过了 softmax, 是概率)
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

# fit 网络训练
model.fit(x_train, y_train,
          batch_size=32, epochs=5,
          validation_data=(x_test, y_test),
          validation_freq=1)

# summary 输出网络结构,参数信息
model.summary()
