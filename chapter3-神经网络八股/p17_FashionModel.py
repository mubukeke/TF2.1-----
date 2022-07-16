from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D


class FashionModel(Model):
    def __init__(self):
        super(FashionModel, self).__init__()   # 下面搭建网络结构
        self.flatten_layer = Flatten()
        self.Dense_layer_128 = Dense(128, activation="relu")
        self.Dense_layer_10 = Dense(10, activation="softmax")

    def call(self, x):
        flatten = self.flatten_layer(x)
        dense_128 = self.Dense_layer_128(flatten)
        y = self.Dense_layer_10(dense_128)
        return y
