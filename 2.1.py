import  tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

#生成样本点shengcheng200哥 200行一列
x_data= np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) +noise
# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义 神经网络中间问题
weights_l1 = tf.Variable(tf.random.normal([1,10]))
biasse_l1 = tf.Variable(tf.zeros([1, 10]))
wx_plus = tf.matmul(x, weights_l1) +biasse_l1
l1 = tf.nn.tanh(wx_plus) #双曲正弦 激活函数
#定义 输出 层 神经网络
weights_l2 = tf.Variable(tf.random_normal([10,1]))
biasse_l2 = tf.Variable(tf.zeros([1,1]))
wx_plus2  = tf.matmul(l1,weights_l2) +biasse_l2
pred = tf.nn.tanh(wx_plus2)
loss = tf.reduce_mean(tf.square(y  - pred))
#使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(20000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    #获得预测值
    pred_value= sess.run(pred, feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,pred_value,'r-', lw = 5)
    plt.show()
