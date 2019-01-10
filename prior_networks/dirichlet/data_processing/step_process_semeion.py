#!/usr/bin/python
import argparse
import os
import sys
import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import FactorAnalysis as FA
from scipy.misc import imresize
import core.utilities.tfrecord_utils  as tfrecord_utils

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('target_path', type=str,
                               help='where to save tfrecords')
commandLineParser.add_argument('--XVAL', type=bool, default=False,
                               help='Whether to process full dataset, or do 10-fold cross validation, by holding out each digit')

def process_data(args):
    data = np.loadtxt(args.data_path)
    data_X = data[:,:256]
    data_y = np.argmax(np.asarray(data[:,256:], dtype=np.int32), axis=1)
    data_X = np.reshape(data_X ,[-1, 16, 16])
    num_examples=data.shape[0]
    print num_examples
    for index in range(num_examples):
        if index % 10000 == 0:
            try:
                writer.close()
            except:
                pass
            writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, 'semeion_' + str(index/10000) + '.tfrecord'))

        image_raw = imresize(data_X[index], size=[28,28]).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tfrecord_utils.int64_feature([28]),
            'width': tfrecord_utils.int64_feature([28]),
            'depth': tfrecord_utils.int64_feature([1]),
            'label': tfrecord_utils.int64_feature([data_y[index]]),
            'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
        writer.write(example.SerializeToString())

    try:
        writer.close()
    except:
        pass

def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_process_semeion_data.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    os.makedirs(args.target_path)
    process_data(args)

    #make_FA(args)
if __name__ == '__main__':
    main()