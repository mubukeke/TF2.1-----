import tensorflow as tf
from p_write_variable_datas import StorageVariableData

# p45_iris.py 中 line 131 行中，比较预测输出与标签是否相等，计算acc.
# 疑惑：因为 pred 和 y_test 都是向量，使用 tf.equal() 是直接按位判断吗？然后记录出一共相同的位数？  NO!
# epoch 0 后在测试集上进行 acc 计算。

# epoch 0 后，使用 w1, b1进行前向传播，进过softmax，计算三个预测值中最大的那个所在的索引
# softmax 后 一个样本输出 [0.2, 0.177, 0.623]
pred = tf.constant(
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                   )
# 测试集中30个样本的类别标签
y_test = tf.constant(
                      [1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 2, 2, 0, 2, 0, 0, 2, 1]
                    )

pred = tf.cast(pred, dtype=y_test.dtype)
# print("pred:\n", pred)
# print("y_test:\n", y_test)
# 实例化类对象，打开文件
# storage_pred = StorageVariableData(pred, "../data/pred.txt")
# storage_y_test = StorageVariableData(y_test, "../data/y_test.txt")

equal_output = tf.equal(pred, y_test)
# print("wether pred equal y_test:\n", output)

correct_num = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)  # 主要是统计相同位有几个，所以求同或也是可以的。
# print("correct_num:\n", correct_num)
"""
pred:
 tf.Tensor([2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2], shape=(30,), dtype=int32)
y_test:
 tf.Tensor([1 0 1 1 1 2 1 1 1 1 0 0 0 0 0 1 1 0 1 0 1 1 2 2 0 2 0 0 2 1], shape=(30,), dtype=int32)
wether pred equal y_test:
 tf.Tensor(
[False False False False False  True False False False False False False
 False False False False False False False False False False  True  True
 False  True False False  True False], shape=(30,), dtype=bool)
correct_num:
 tf.Tensor([0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0], shape=(30,), dtype=int32)
"""

correct_number = tf.reduce_sum(correct_num)   # 将相同的位数累加起来，得到预测输出与测试标签 相同的位数
# print("correct number:\n", correct_number)    # 该例子中有 5 个相同位

# 所有变量数据写入文件
storage_pred = StorageVariableData(pred, "../data/pred.txt")
storage_y_test = StorageVariableData(y_test, "../data/y_test.txt")
storage_output = StorageVariableData(equal_output, "../data/equal_output.txt")
storage_correct_num = StorageVariableData(correct_num, "../data/correct_number.txt")

storage_pred.open_write_variable_data()
storage_y_test.open_write_variable_data()
storage_output.open_write_variable_data()
storage_correct_num.open_write_variable_data()

# 文件关闭
storage_pred.close_file()
storage_y_test.close_file()
storage_output.close_file()
storage_correct_num.close_file()

