#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/12 14:00
#@Author: jccc
#@File  : 10.2生成tfrecord文件.py
import tensorflow as tf
import  os
import random
import math
import sys
from PIL import Image
import numpy as np
#验证集数量
_num_test = 500
#随机种子
_random_seed= 0
#数据集路径
DATASET_DIR ='E:/bili_tensorflow_test/captcha/images/'
#生成tfrecord
TFRECORD_DIR='E:/bili_tensorflow_test/captcha/'

#判断tfrecord 文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(dataset_dir, split_name+'tfrecords')
        if not tf.gfile.Exists(output_filename):
            return  False
        return  True
#获取所有验证码图片
def _get_filename_and_classes(dataset_dir):
    photo_names =[]
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path = os.path.join(dataset_dir,filename)
        photo_names.append(path)
    return  photo_names

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values =[values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
def image_to_tfample(image_data, label0,label1,label2,label3):
    return tf.train.Example(features = tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label0':int64_feature(label0),
        'label1': int64_feature(label1),
        'label2': int64_feature(label2),
        'label3': int64_feature(label3),
    }))

def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']

    with tf.Session() as sess:
        #定义tfrecord 文件路径+名字
        output_filename = os.path.join(dataset_dir, split_name +'.tfrecord')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i, filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converint image %d/%d ' % (i+1,len(filenames)))
                    sys.stdout.flush()

                    #读取图片
                    image_data = Image.open(filename)
                    #根据模型的结构resize
                    image_data = image_data.resize((224,224))
                    #灰度化
                    image_data = np.array(image_data.convert('L'))
                    #将图片幻化为bytes
                    image_data = image_data.tobytes()

                    #获取label
                    labels = filename.split('/')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))

                    #生成 protocol 数据类型
                    example = image_to_tfample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                except IOError as e:
                    print('Could not read:', filename)
                    print('Error', e)
                    print('Skip it\n.')

if _dataset_exists(TFRECORD_DIR):
    print('tfrecord 文件已经存在')
else:
    #获取所有图片
    photo_filenames = _get_filename_and_classes(DATASET_DIR)
    #把数据切分为训练集和测试集
    random.seed(_random_seed)
    random.shuffle(photo_filenames)
    training_filenames =photo_filenames[_num_test:]
    testing_filenames = photo_filenames[:_num_test]

    #数据转换
    _convert_dataset('train', training_filenames, TFRECORD_DIR)
    _convert_dataset('test', testing_filenames, TFRECORD_DIR)
    print('\n 生成tfcecord文件')

