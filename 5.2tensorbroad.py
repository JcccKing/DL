#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/10 21:09
#@Author: jccc
#@File  : 5.2tensorbroad.py

  #!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/10 17:04
#@Author: jccc
#@File  : softmax.py
#  网络运行
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
from tensorflow.contrib.tensorboard.plugins import projector
#载入数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义批次大小
batch_size = 100
#一共多少批次
n_batch = mnist.train.num_examples // batch_size

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) #平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev) #标准差
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var) #直方图
#命名空间
with tf.name_scope('input'):
# 定义两个placeholder
    x  = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y  = tf.placeholder(tf.float32, [None, 10], name='y-input')
with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        #创建一个 简单的神经网络
        w = tf.Variable(tf.zeros([784,10]), name='W')
        variable_summaries(w)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, w) + b
    with tf.name_scope('softmax'):
        pre = tf.nn.softmax(wx_plus_b)

#定义 二次代价函数
#loss = tf.reduce_mean(tf.square(y - pre))
with tf.name_scope('loss'):
# 交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pre))
    tf.summary.scalar('loss',loss)

#梯度下降
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train_step = tf.train.AdadeltaOptimizer(0.2).minimize(loss)
#chushihua
init = tf.global_variables_initializer()
# 求标签最大的值在哪个位置 结果存放在一个 布尔类型列表 ，#argmax最大值所在文职
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pre, 1))
with tf.name_scope('taccuracy'):
    #求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

#合并所有的 summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x:batch_xs, y:batch_ys})

        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("after "+str(epoch) +"test accuracy" + str(acc))