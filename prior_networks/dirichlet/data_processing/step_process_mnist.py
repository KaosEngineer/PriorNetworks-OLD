#!/usr/bin/python
import argparse
import os
import sys

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

import core.utilities.tfrecord_utils  as tfrecord_utils

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('target_path', type=str,
                               help='where to save tfrecords')
commandLineParser.add_argument('--XVAL', type=bool, default=False,
                               help='Whether to process full dataset, or do 10-fold cross validation, by holding out each digit')

def process_data(data, args):
    sets = ['train', 'valid', 'test']
    for set in sets:
        images = data[set].images
        labels = data[set].labels
        num_examples = data[set].num_examples

        for index in range(num_examples):
            label = int(labels[index])
            if index % 10000 == 0:
                try:
                    writer.close()
                except:
                    pass
                writer = tf.python_io.TFRecordWriter(
                    os.path.join(args.target_path, set + '_' + str(index / 10000) + '.tfrecord'))

            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tfrecord_utils.int64_feature([28]),
                'width': tfrecord_utils.int64_feature([28]),
                'depth': tfrecord_utils.int64_feature([1]),
                'label': tfrecord_utils.int64_feature([label]),
                'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
            writer.write(example.SerializeToString())
    try:
        writer.close()
    except:
        pass

def process_data_XVAL(data, args):
    sets = ['train', 'valid', 'test']
    for set in sets:
        images = data[set].images
        labels = data[set].labels
        num_examples = data[set].num_examples

        for fold in xrange(10):
            print 'Fold', fold
            name = 'fold_'+str(fold)
            path = os.path.join(args.target_path, name)
            if not os.path.isdir(path):
                os.makedirs(path)
            for index in range(num_examples):
                label = int(labels[index])
                if index % 10000 == 0:
                    try:
                        writer_seen.close()
                        writer_unseen.close()
                    except:
                        pass
                    writer_seen = tf.python_io.TFRecordWriter(os.path.join(path,  set+'_'+name + '_'+str(index/10000)+'.tfrecord'))
                    writer_unseen = tf.python_io.TFRecordWriter(os.path.join(path, set + '_heldout_' +  name + '_' + str(index / 10000) + '.tfrecord'))

                if label > fold:
                    label -= 1
                elif label == fold:
                    label = 9
                image_raw = images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tfrecord_utils.int64_feature([28]),
                    'width': tfrecord_utils.int64_feature([28]),
                    'depth': tfrecord_utils.int64_feature([1]),
                    'label': tfrecord_utils.int64_feature([label]),
                    'image_raw': tfrecord_utils.bytes_feature([image_raw])}))

                if int(labels[index]) == fold:
                    writer_unseen.write(example.SerializeToString())
                else:
                    writer_seen.write(example.SerializeToString())
    try:
        writer_seen.close()
        writer_unseen.close()
    except:
        pass

def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_process_mnist_data.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if not os.path.isdir(args.target_path):
        os.makedirs(args.target_path)
    data = mnist.read_data_sets(args.data_path, dtype=tf.uint8)

    data = {'train' : data.train, 'valid' : data.validation, 'test' : data.test}


    if args.XVAL == True:
        process_data_XVAL(data, args)
    else:
        process_data(data, args)


if __name__ == '__main__':
    main()