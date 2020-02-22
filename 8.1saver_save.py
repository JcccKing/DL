#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/11 15:19
#@Author: jccc
#@File  : 8.1saver_save.py

import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
#载入数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#定义批次大小
batch_size = 100
#一共多少批次
n_batch = mnist.train.num_examples // batch_size
# 定义两个placeholder
x  = tf.placeholder(tf.float32, [None, 784])
y  = tf.placeholder(tf.float32, [None, 10])

#创建一个 简单的神经网络
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
pre = tf.nn.softmax(tf.matmul(x, w)+b)

#定义 二次代价函数
#loss = tf.reduce_mean(tf.square(y - pre))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pre))
#梯度下降
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train_step = tf.train.AdadeltaOptimizer(0.2).minimize(loss)
#chushihua
init = tf.global_variables_initializer()
# 求标签最大的值在哪个位置 结果存放在一个 布尔类型列表 ，#argmax最大值所在文职
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pre, 1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    saver.restore(sess, 'net/mode.ckpt')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    # for epoch in range(11):
    #     for batch in range(n_batch):
    #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #         sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
    #
    #     acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    #     print("after "+str(epoch) +"test accuracy" + str(acc))
    #     # 保存的模型结构和 参数
    #     saver.save(sess, 'net/mode.ckpt')
