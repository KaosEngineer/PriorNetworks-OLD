#!/usr/bin/python
import argparse
import os
import sys
import matplotlib

#matplotlib.use('agg')

from matplotlib import pyplot as plt
from PIL import Image

import numpy as np
import tensorflow as tf

import core.utilities.tfrecord_utils  as tfrecord_utils

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('target_path', type=str,
                               help='where to save tfrecords')
commandLineParser.add_argument('size', type=int,
                               help='where to save tfrecords')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_process_omniglot_data.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if not os.path.isdir(args.target_path):
        os.makedirs(args.target_path)

    collage = np.zeros(shape=[4 * args.size, 8 * args.size])
    dirs = os.listdir(args.data_path)
    len_dirs = len(dirs)
    for item, j in zip(dirs, xrange(len_dirs)):
        if j % 15000 == 0:
            try:
                writer.close()
            except:
                pass
            writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, 'omniglot_' + str(j / 15000) + '.tfrecord'))
            print j
        img_file = os.path.join(args.data_path, item)
        if os.path.isfile(img_file) and os.stat(img_file).st_size != 0:
            try:
                im = Image.open(img_file)
                width, height = im.size
                size = np.min([width, height])
                if size < args.size:
                    continue
                imResize = im.resize((args.size, args.size), resample=Image.NEAREST)
                imResize=np.array(imResize.getdata(), dtype=np.uint8).reshape(args.size, args.size)
                if j < 32:
                    i = j % 8
                    k = j / 8
                    collage[k * args.size:(k + 1) * args.size, i * args.size:(i + 1) * args.size] = imResize
                elif j == 32:
                    fig = plt.imshow(np.asarray(collage, dtype=np.uint8), cmap='gray')
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    path = os.path.join(args.target_path, 'omniglot.png')
                    plt.savefig(path, bbox_inches='tight')
                    plt.close()

                imResize=np.reshape(imResize, (args.size*args.size))
                imResize_raw = imResize.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tfrecord_utils.int64_feature([args.size]),
                    'width': tfrecord_utils.int64_feature([args.size]),
                    'depth': tfrecord_utils.int64_feature([1]),
                    'label': tfrecord_utils.int64_feature([-1]),
                    'image_raw': tfrecord_utils.bytes_feature([imResize_raw])}))
                writer.write(example.SerializeToString())
            except:
                with open('errors', 'a') as handle:
                    handle.write(item + '\n')
                    print 'here'
    writer.close()
