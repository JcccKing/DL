#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/11 12:05
#@Author: jccc
#@File  : 7.1卷积神经网络Mnist可视化.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
batch_size = 100
n_batch  = mnist.train.num_examples // batch_size

#参数概要
def variable_summaries(var):
    with tf.name_scope('symmaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) #平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev) # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram'. var) #直方图
#初始化 权值
def weight_variable(shape,name):
    inital = tf.truncated_normal(shape, stddev=0.1) #生成一个正态分布
    return tf.Variable(inital,name)
# 初始化偏置值
def bias_variable(shape,name):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)
#卷积层
def conv2d(x,w):
    # x是形状 [batch,chang,kuan,通道数]
    # w
    # 步长[1,x方向，y方向 1]
    # same valid
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
#池化层
def max_pool_2x2(x):
    #最大池化
    # x是形状 [batch,chang,kuan,通道数]
    #ksize 窗口大小 [1,x方向，y方向 1]

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('input'):
    #定义两个placeholde
    x = tf.placeholder(tf.float32, [None, 784]) # 28*28
    y = tf.placeholder(tf.float32, [None, 10])  #0-9 十个数字输出
    with tf.name_scope('x-image'):
        #改变 x 的 格式转化成 4d [batch,chang,kuan, 通道数 1维 彩色 就是 3
        x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('Conv1'):
    with tf.name_scope('w_conv1'):
        #初始化 第一个 卷积层权值 偏置值
        w_conv1 = weight_variable([5, 5, 1, 32],name='w_conv1') #5*5的采样窗口 32 个卷积核 从 1 平面抽取
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32],name='b_conv1')#每个 j卷积核一个 偏置值

    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, w_conv1) + b_conv1
        #把x——image 和权值进行卷积 ，再加上偏置值 最大池化，用于激活函数 得到 32 平面
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('max_pool_1'):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('Conv2'):
    with tf.name_scope('w_conv2'):
        #初始化第二个 卷积层 权值和偏置值
        w_conv2 = weight_variable([5, 5, 32, 64], name='w_conv2') #5*5的采样 63 卷积核 从 32平面取
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')#每个 j卷积核一个 偏置值
    with tf.name_scope('conv2d_1'):
        conv2d_2 = conv2d(h_pool1, w_conv2) + b_conv2
    with tf.name_scope('relu'):
        #把hpool1 进行卷积 ，再加上偏置值,最大池化
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('max_pool_2'):
        h_pool2 = max_pool_2x2(h_conv2)

#卷积 不改变 大小 因为 padding = same
# 28*28 第一次卷积后 还是 28*28 池化是2*2 就变成14*14
#二次卷积 变成 14*14  池化后 7*7
#从上面 两次 卷积 池化后得到 64张 7*7的 平面

#初始化第一个全连接层的权值
with tf.name_scope('fc1'):
    with tf.name_scope('w_fc1'):
        w_fc1 = weight_variable([7*7*64, 1024],name='w_fc1') #上一层有 7*7*64个神经元 ，全连接层共 1024个 神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024],name='b_fc1')

with tf.name_scope('h_pool2_flat'):
#把池化层2 的输出层扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')
with tf.name_scope('relu'):
    #求第一个全连接层 输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)
with tf.name_scope('keep_prob'):
    #keep_prob 用来表示神经元的输出 概率 先占位
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
with tf.name_scope('h_fc1_drop'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name='h_fc1_drop')

with tf.name_scope('fc2'):
    with tf.name_scope('w_fc2'):
        #初始化第二个 全连接层
        w_fc2 = weight_variable([1024,10], name='w_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope('soft_max'):
        #计算输出
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)
with tf.name_scope('cross_entropy'):
    #交叉熵 代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    #使用优化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuaracy'):
    with tf.name_scope('correct_prediction'):
        #结果存入 bool 列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuaracy'):
        #求准去率
        accuaracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuaracy', accuaracy)
#合并所有
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 0.5})
        #记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x:batch_xs, y :batch_ys,keep_prob:1.0})
        train_writer.add_summary(summary, i)
        #记录测试机
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i %100 == 0:
            test_acc = sess.run(accuaracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            train_acc = sess.run(accuaracy, feed_dict={x:mnist.train.images[:10000], y:mnist.train.labels[:10000],keep_prob:1.0})

            print("after "+str(i) +" test accuaray = "+str(test_acc)+' train accuracy = '+str(train_acc))
        # after 1000 test accuaray = 0.8749 train accuracy = 0.8739