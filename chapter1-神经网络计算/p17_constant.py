import tensorflow as tf

# tf.constant(张量内容，dtype=数据类型（可选）)
a = tf.constant([1, 5, 3], dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape)  # (3,) 看逗号间隔几个数，张量就是几维的。3 表示一维张量有三个元素。
# shape = (2, 3, 4) 逗号间隔三个数，三维张量