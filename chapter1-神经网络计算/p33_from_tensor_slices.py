import tensorflow as tf

# tf.data.Dataset.from_tensor_slices((features, labels))
# 可以将相同长度的 （特征，标签）构成这样的数据对

features = tf.constant([[12, 23, 20, 17],
                       [30, 40, 50, 43],
                       [32, 44, 52, 38],
                       [11, 9, 10, 21]])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)

for element in dataset:
    print(element)
