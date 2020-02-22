#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/10 20:29
#@Author: jccc
#@File  : 4.3Optimizer.py
#优化器对比
#  标准 随机 批量 w：要训练的参数，j(w)代价函数 代价函数梯度
# tf.train.AdadeltaOptimizer
# tf.train.GradientDescentOptimizer
# tf.train.MomentumOptimizer
# tf.train.RMSPropOptimizer

#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/10 19:52
#@Author: jccc
#@File  : dropout.py
# 拟合 和交叉熵
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
#载入数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#定义批次大小
batch_size = 100
#一共多少批次
n_batch = mnist.train.num_examples // batch_size
# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001,tf.float32)

#创建一个 简单的神经网络
w1 = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))#标准差
b1 = tf.Variable(tf.zeros([100]) +0.1)
L1 = tf.nn.tanh(tf.matmul(x,w1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)
#
# w2= tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))#标准差
# b2= tf.Variable(tf.zeros([2000]) +0.1)
# L2 = tf.nn.tanh(tf.matmul(L1_drop,w2) + b2)
# L2_drop = tf.nn.dropout(L2, keep_prob)

w3= tf.Variable(tf.truncated_normal([100, 50], stddev=0.1))#标准差
b3= tf.Variable(tf.zeros([50]) +0.1)
L3 = tf.nn.tanh(tf.matmul(L1_drop,w3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

# w = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
w4 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))#标准差
b4 = tf.Variable(tf.zeros([10]) +0.1)
pre = tf.nn.softmax(tf.matmul(L3_drop, w4)+ b4)

#定义 二次代价函数
#loss = tf.reduce_mean(tf.square(y - pre))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pre))
#梯度下降
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
#rain_step = tf.train.AdadeltaOptimizer(0.01).minimize(loss)
#chushihua
init = tf.global_variables_initializer()
# 求标签最大的值在哪个位置 结果存放在一个 布尔类型列表 ，#argmax最大值所在文职
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pre, 1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))  #类似于钟摆运动 的 学习率
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.8})

        test_cc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1.0})
        train_cc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("after "+str(epoch) +" test accuracy " + str(test_cc)+"  accuracy " +str(train_cc))