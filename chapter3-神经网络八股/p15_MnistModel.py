from tensorflow.keras import Model  # 导入基类"模块
from tensorflow.keras.layers import Dense, Conv2D, Flatten   # 导入相应网络层模块
import tensorflow as tf

class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten_layer = Flatten()
        self.Dense_layer_128 = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2())
        self.Dense_layer_10 = Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        hidden_layer_1 = self.flatten_layer(x)
        hidden_layer_2 = self.Dense_layer_128(hidden_layer_1)
        y = self.Dense_layer_10(hidden_layer_2)
        return y
