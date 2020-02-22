#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/2/12 11:19
#@Author: jccc
#@File  : 9.2生成tfrecord.py
import tensorflow as tf
import  os
import  random
import  math
import  sys

#验证集数量
num_test = 500
#随机种子
_random_seed = 0
#数据块
num_shards =5
#数据集路径
dataset_dir = ''
#标签文件
LABELS_FILENAME= ''

#定义 tfrecord 文件的路径和名字
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'image_%s_%05d-of-%05d.tfrecord' %(split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)

#判断tfrecord 文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(num_shards):
            #定义tfrecord 文件路径和名字
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        if not tf.gfile.Exists(output_filename):
            return  False
    return  True

#获取文件所有分类
def _get_filenames_and_classes(dataset_dir):
    #数据目录
    directories =[]
    #分类名称
    class_names =[]
    for filename in os.listdir(dataset_dir):
        #合并文件路径
        path = os.path.join(dataset_dir, filename)
        #判断该路径是否为目录
        if os.path.isdir(path):
            #加入数据目录：
            directories.append(path)
            #加入类别名称
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            #把图片加入图片列表
            photo_filenames.append(path)

    return photo_filenames, class_names

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list = tf.train.Int64List(values))

def bytes_feature(values):
    return  tf.train.Feature(bytes_list = tf.train.BytesList(values))

def image_to_tfexample(image_data,image_format, class_id):
    return  tf.train.Example(features = tf.train.Features(feature = {
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
    }))

def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

#把数据转化为TFrecord 格式
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'test']
    #计算每个数据块有多少数据
    num_per_shard = int(len(filenames) / num_shards)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(num_shards):
                #定义 tfrecord 文件路径和名字
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    #每个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    #每个数据快最后的位置
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx,end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                            sys.stdout.flush()
                            #读取图片
                            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                            #获得图片的类别名称
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            #找到类别 名称对应id
                            class_id = class_names_to_ids[class_name]
                            #生成tfrecord 文件
                            example = image_to_tfexample(image_data, b'jpg', class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print('Could not read:', filenames[i])
                            print('Error', e)
                            print('Skip it\n.')

    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__=='__main__':
    #判断tfrecord 文件是否存在
    if _dataset_exists(dataset_dir):
        print('tfrecord 已经存在')
    else:
        #获得所有图片以及分类
        photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
        #把分类转化为字典格式，类似于{'house' : 3 ,'flower' : 1, 'plane' : 4,}
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        #把数据切分成训练集和测试集
        random.seed(_random_seed)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[num_test:]
        testing_filenames = photo_filenames[:num_test]

        #数据转换
        _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)
        _convert_dataset('test', testing_filenames, class_names_to_ids, dataset_dir)

        #输出 labels 文件
        label_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(label_to_class_names, dataset_dir)






