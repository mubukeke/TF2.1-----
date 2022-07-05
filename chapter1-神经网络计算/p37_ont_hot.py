import tensorflow as tf

# ont-hot 形式的独热码，可以将标签生成相应的矩阵，表示某一个标签在多分类中，属于的那个类为1，其他类为0

classes = 4
labels = tf.constant([1, 2, 0, 0])
output = tf.one_hot(labels, depth=classes)
print(output)
