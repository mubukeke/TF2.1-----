import tensorflow as tf
# 交叉熵是用来衡量 两组概率分布结果和标准概率分布 之间的距离。

y_ = tf.constant([1, 0], dtype=tf.float32)
y1 = tf.constant([0.6, 0.4], dtype=tf.float32)   # 预测输出概率1
y2 = tf.constant([0.8, 0.2], dtype=tf.float32)   # 预测输出概率2

loss_ce1 = tf.losses.categorical_crossentropy(y_, y1)
loss_ce2 = tf.losses.categorical_crossentropy(y_, y2)
print("loss_ce1:", loss_ce1)
print("loss_ce2:", loss_ce2)

loss_formular_ce1 = 0
loss_formular_ce2 = 0
# 验证ce公式   Loss_cross_entropy = -求和（y_label_i * log(y_pred_i)）
for index, ele in enumerate(y_.numpy()):
    print("ele_y_: %f, ele_y1: %f, ele_y2: %f." %(y_[index], y1[index], y2[index]))
    loss_formular_ce1 += y_[index]*tf.math.log(y1[index])
    loss_formular_ce2 += y_[index]*tf.math.log(y2[index])
print("loss_formular_ce1: ", -loss_formular_ce1)
print("loss_formular_ce2: ", -loss_formular_ce2)

# for ele_y_ in y_:
#     print("ele_y_: %f" %(ele_y_))


# print("Loss_ce1 formular:", -1*tf.math.log(0.6))
# print("Loss_ce2 formular:", -1*tf.math.log(0.8))

# 交叉熵损失函数