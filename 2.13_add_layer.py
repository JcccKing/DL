#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/13 21:06
#@Author: jccc
#@File  : 2.13_add_layer.py
import tensorflow as tf
import  numpy as np
def add_layer(inputs, in_size, out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biase = tf.Variable(tf.zeros([1,out_size]) +0.1)
    wx_plus_b = tf.matmul(inputs,Weights) +biase
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return  outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) -0.5
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
l1 =add_layer(xs,1,10,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):

    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    if i % 50:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
