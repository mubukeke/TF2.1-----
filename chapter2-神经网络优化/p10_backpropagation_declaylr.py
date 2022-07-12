import tensorflow as tf
from matplotlib import pyplot as plt

w = tf.Variable(tf.constant(5, dtype=tf.float32))

epoch = 40
LR_BASE = 0.2   # 最初学习率
LR_DECAY = 0.8  # 学习率衰减率
LR_STEP = 2   # 喂入多少轮后，更新一次学习率
lr_values = []
test = []

for epoch in range(epoch):
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    lr_values.append(lr)
    test.append(epoch / LR_STEP)
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)

    grads = tape.gradient(loss, w)

    w.assign_sub(lr*grads)
    print("After %i epoch, w is %f, loss is %f, lr is %f" % (epoch, w.numpy(), loss, lr))

# 查看 epoch / LR_STEP 值。
print("test: ", test)

# 添加 x 轴 1:40:1
x = []
for i in range(1, 41, 1):
    x.append(i)
print(x)
plt.scatter(x, lr_values, marker="o", color='r')
plt.show()
