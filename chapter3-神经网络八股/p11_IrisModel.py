import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense  # 那我要使用卷积层哩？就import卷积？
from tensorflow.keras.layers import Conv2D  # 还真的是啊！？？？没验证


class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.dense_lay = Dense(3,
                               activation="softmax",
                               kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.dense_lay(x)
        return y

    def print(self):
       print("test whether can create class object.")