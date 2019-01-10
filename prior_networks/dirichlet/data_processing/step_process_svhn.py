#!/usr/bin/python
import argparse
import os
import sys

import matplotlib

matplotlib.use('agg')

from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.decomposition import FactorAnalysis as FA
from scipy.misc import imresize
import core.utilities.tfrecord_utils  as tfrecord_utils

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('target_path', type=str,
                               help='where to save tfrecords')
commandLineParser.add_argument('--greyscale', type=bool, default=False,
                               help='Make greyscale version')
commandLineParser.add_argument('--size', type=int, default=32,
                               help='Specify size for resizing')
commandLineParser.add_argument('--XVAL', type=bool, default=False,
                               help='Whether to process full dataset, or do 10-fold cross validation, by holding out each digit')

def load_svhn_mat(path):
    data = sio.loadmat(path)
    data_y = data['y']
    data_y = np.squeeze(data_y)
    data_X = data['X']
    data_X = np.transpose(data_X, axes=(3, 0, 1, 2))

    data_y[data_y[:] == 10] = 0
    return data_y, data_X



def process_data_greyscale(args):
    files=['test_32x32.mat', 'train_32x32.mat', 'extra_32x32.mat', ]

    for file in files:
        data_y, data_X = load_svhn_mat(os.path.join(args.data_path,file))
        data_X = 0.21*data_X[:,:,:,0]+0.72*data_X[:,:,:,1]+0.07*data_X[:,:,:,2]
        data_X = np.asarray(data_X, dtype=np.int32)
        print data_X.shape
        fname=file.split('.')[0]

        num_examples=data_X.shape[0]
        print num_examples
        for index in range(num_examples):
            label = int(data_y[index])
            if index % 10000 == 0:
                try:
                    writer.close()
                except:
                    pass
                writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, fname + '_gs_' + str(index/10000) + '.tfrecord'))

            image_raw = imresize(data_X[index], [args.size,args.size]).tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tfrecord_utils.int64_feature([args.size]),
                'width': tfrecord_utils.int64_feature([args.size]),
                'depth': tfrecord_utils.int64_feature([1]),
                'label': tfrecord_utils.int64_feature([label]),
                'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
            writer.write(example.SerializeToString())

    try:
        writer.close()
    except:
        pass

def process_data(args):
    files=['test_32x32.mat', 'train_32x32.mat', 'extra_32x32.mat', ]

    for file in files:
        data_y, data_X = load_svhn_mat(os.path.join(args.data_path,file))
        fname=file.split('.')[0]

        num_examples=data_X.shape[0]
        print num_examples
        for index in range(num_examples):
            label = int(data_y[index])
            if index % 10000 == 0:
                try:
                    writer.close()
                except:
                    pass
                writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, fname + '_' + str(index/10000) + '.tfrecord'))


            image_raw = data_X[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tfrecord_utils.int64_feature([32]),
                'width': tfrecord_utils.int64_feature([32]),
                'depth': tfrecord_utils.int64_feature([1]),
                'label': tfrecord_utils.int64_feature([label]),
                'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
            writer.write(example.SerializeToString())

    try:
        writer.close()
    except:
        pass

def process_data_XVAL(args):

    files=['test_32x32.mat', 'train_32x32.mat', 'extra_32x32.mat', ]

    for file in files:
        data_y, data_X = load_svhn_mat(os.path.join(args.data_path,file))
        set=file.split('.')[0]
        max_digits=np.max(data_y)

        num_examples=data_X.shape[0]
        for fold in range(max_digits+1):
            print 'Fold', fold
            fname='fold_'+str(fold)
            path = os.path.join(args.target_path, fname)
            if not os.path.isdir(path):
                os.makedirs(path)
            for index in range(num_examples):
                label = int(data_y[index])
                if label > fold:
                    label -= 1
                elif label == fold:
                    label = max_digits
                if index % 42660 == 0:
                    try:
                        writer_seen.close()
                        writer_unseen.close()
                    except:
                        pass
                    writer_seen = tf.python_io.TFRecordWriter(os.path.join(path,  set+'_'+fname + '_'+str(index/42660)+'.tfrecord'))
                    writer_unseen = tf.python_io.TFRecordWriter(os.path.join(path, set + '_heldout_' +  fname + '_' + str(index / 42660) + '.tfrecord'))

                image_raw = data_X[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tfrecord_utils.int64_feature([32]),
                    'width': tfrecord_utils.int64_feature([32]),
                    'depth': tfrecord_utils.int64_feature([3]),
                    'label': tfrecord_utils.int64_feature([label]),
                    'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
                if int(data_y[index]) == fold:
                    writer_unseen.write(example.SerializeToString())
                else:
                    writer_seen.write(example.SerializeToString())

    try:
        writer_seen.close()
        writer_unseen.close()
    except:
        pass

def make_FA(args):
    data_y, data_X = load_svhn_mat(os.path.join(args.data_path,'train_32x32.mat'))
    components=200
    data_X = (np.reshape(data_X, (-1, 3072)) - 127) / 128.0
    model = FA(n_components=components,
               tol = 0.01,
               copy = True,
               max_iter = 400,
               noise_variance_init = None,
               svd_method = 'randomized',
               iterated_power = 3,
               random_state = 0)
    model.fit(data_X)
    mean = model.mean_
    variance = model.noise_variance_
    load_matrix = model.components_

    path = os.path.join(args.target_path, "whole")
    os.makedirs(path)
    np.savetxt(os.path.join(path, "fa_mean.txt"), mean)
    np.savetxt(os.path.join(path, "fa_variance.txt"), variance)
    np.savetxt(os.path.join(path, "fa_loading_matrix.txt"), load_matrix)

    samples = 32
    noise = 1
    z = np.random.multivariate_normal(np.zeros(shape=[components]), noise * np.eye(components), samples)
    x = np.random.multivariate_normal(np.zeros(shape=[3072]), np.eye(3072), samples)

    mu = np.dot(z, load_matrix) + mean
    images = (mu + np.sqrt(noise * variance) * x) * 128 + 127
    images[images>255.0]=255.0
    images[images<0.0]=0.0
    collage = np.zeros(shape=[4 * 32, 8 * 32, 3])
    for i in xrange(32):
        j = i % 8
        k = i / 8
        img = images[i].reshape([32, 32,3])
        collage[k * 32:(k + 1) * 32, j * 32:(j + 1) * 32, :] = img

    fig = plt.imshow(np.asarray(collage, dtype=np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(path,'fa_noise_'+str(noise)+'.png'), bbox_inches='tight')
    #plt.show()
    plt.close()

    collage = np.zeros(shape=[4 * 32, 8 * 32, 3])
    for i in xrange(32):
        j = i % 8
        k = i / 8
        img = data_X[i].reshape([32, 32, 3])*128+127
        collage[k * 32:(k + 1) * 32, j * 32:(j + 1) * 32, :] = img

    fig = plt.imshow(np.asarray(collage, dtype=np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(path, 'svhn_samples.png'), bbox_inches='tight')
    plt.close()

def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_process_svhn_data.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    os.makedirs(args.target_path)
    if args.XVAL == True:
        process_data_XVAL(args)
    elif args.greyscale == True:
        process_data_greyscale(args)
    else:
        process_data(args)


    #make_FA(args)
if __name__ == '__main__':
    main()