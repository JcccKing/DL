#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/11 13:22
#@Author: jccc
#@File  : RNN.py
#rnn 递归循环网络 LSTM
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

n_inputs = 28 #输入一行，一行28个数据
max_time = 28 #一共28行
lstm_size = 100 #随层单元
n_classes = 10 #10个单元
batch_size = 50
n_batch  = mnist.train.num_examples // batch_size

#这里的none表示第一个维度可以是任意长度
x = tf.placeholder(tf.float32, [None, 784])
#正确的标签
y = tf.placeholder(tf.float32, [None, 10])
# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
#初始化 偏置值
biases =tf.Variable(tf.constant(0.1, shape=[n_classes]))

#定义rnn 网络
def RNN(X,weights,biases):
    #input s  = batch_size, maxtime,n_inputs
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    #定义基本的单元 cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    #final_state[0] 是cell state
    #final_state[1] 是hidden_state  输出
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results

#计算rnn 返回的结果
prediction =RNN(x, weights, biases)
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
#使用优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个 bool
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('after '+str(epoch)+'accuracy = '+str(acc))
        # 92.3
