#!/usr/bin/python
import argparse
import os
import sys

import matplotlib
import random
matplotlib.use('agg')

from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
import tensorflow as tf
from sklearn.decomposition import FactorAnalysis as FA
from scipy.misc import imresize
import core.utilities.tfrecord_utils  as tfrecord_utils

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('target_path', type=str,
                               help='where to save tfrecords')
commandLineParser.add_argument('--size', type=int, default=32,
                               help='Specify size for resizing')
commandLineParser.add_argument('--XVAL', type=bool, default=False,
                               help='Whether to process full dataset, or do 10-fold cross validation, by holding out each digit')

def process_data(args):
    files = os.listdir(os.path.join(args.data_path,'images'))
    num_examples=len(files)
    #Make sure there is a good global shuffle
    random.shuffle(files)
    for file, i in zip(files, range(num_examples)):
        file_path = os.path.join(args.data_path, 'images/'+file)
        if i % 5000 == 0:
            print i
            try:
                writer.close()
            except:
                pass
            writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, 'LSUN' + '_' + str(i/5000) + '.tfrecord'))
        if os.path.isfile(file_path) and os.stat(file_path).st_size != 0:
            im = Image.open(file_path)
            im = im.resize((args.size, args.size), resample=Image.BICUBIC)
            image_raw = im.convert("RGB").tostring("raw", "RGB")

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tfrecord_utils.int64_feature([args.size]),
                'width': tfrecord_utils.int64_feature([args.size]),
                'depth': tfrecord_utils.int64_feature([3]),
                'label': tfrecord_utils.int64_feature([-1]),
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
    with open('CMDs/step_process_LSUN_data.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    os.makedirs(args.target_path)
    process_data(args)


if __name__ == '__main__':
    main()