#!/usr/bin/python
import argparse
import os
import sys
import matplotlib

matplotlib.use('agg')

from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
from sklearn.decomposition import FactorAnalysis as FA
import core.utilities.tfrecord_utils  as tfrecord_utils

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('target_path', type=str,
                               help='where to save tfrecords')
commandLineParser.add_argument('--makeFA', type=bool, default=False,
                               help='process cifar-100 instead')
commandLineParser.add_argument('--cifar100', type=bool, default=False,
                               help='process cifar-100 instead')
commandLineParser.add_argument('--greyscale', type=bool, default=False,
                               help='Make greyscale version')
commandLineParser.add_argument('--size', type=int, default=32,
                               help='Specify size for resizing')
commandLineParser.add_argument('--XVAL', type=bool, default=False,
                               help='Whether to process full dataset, or do 10-fold cross validation, by holding out each digit')

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        data = cPickle.load(fo)

    try:
        labels = data['labels']
    except:
        labels = data['fine_labels']
    images = data['data']
    print images.shape
    images = np.reshape(images, (-1,3,32,32))
    images = np.transpose(images, [0,2,3,1])
    #images = images[:,:,:,[2,1,0]]
    return labels, images

def process_data(args):
    if args.cifar100 == True:
        files = ['test', 'train']
    else:
        files=['data_batch_1',  'data_batch_2',  'data_batch_3',  'data_batch_4',  'data_batch_5',  'test_batch']

    for file in files:
        labels, images = unpickle(os.path.join(args.data_path,file))
        num_examples=images.shape[0]
        print num_examples

        for index in range(num_examples):
            label = int(labels[index])
            if index % 10000 == 0:
                print index
                try:
                    writer.close()
                except:
                    pass
                writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, file + '_'+ str(index/10000)+'.tfrecord'))
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tfrecord_utils.int64_feature([32]),
                'width': tfrecord_utils.int64_feature([32]),
                'depth': tfrecord_utils.int64_feature([3]),
                'label': tfrecord_utils.int64_feature([label]),
                'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
            writer.write(example.SerializeToString())

    try:
        writer.close()
    except:
        pass

def process_data_greyscale(args):
    files=['data_batch_1',  'data_batch_2',  'data_batch_3',  'data_batch_4',  'data_batch_5',  'test_batch']

    for file in files:
        labels, images = unpickle(os.path.join(args.data_path,file))
        num_examples=images.shape[0]
        images = 0.21 * images[:, :, :, 0] + 0.72 * images[:, :, :, 1] + 0.07 * images[:, :, :, 2]
        images = np.asarray(images, dtype=np.int32)
        print num_examples
        try:
            writer.close()
        except:
            pass
        writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, file + '.tfrecord'))

        for index in range(num_examples):
            label = int(labels[index])

            image_raw = imresize(images[index], [args.size, args.size]).tostring()
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


def process_data_XVAL(args):
    files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

    for file in files:
        labels, images = unpickle(os.path.join(args.data_path,file))
        max_digits=np.max(labels)

        num_examples=images.shape[0]
        for fold in range(max_digits+1):
            print 'Fold', fold
            fname='fold_'+str(fold)
            path = os.path.join(args.target_path, fname)
            if not os.path.isdir(path):
                os.makedirs(path)
            for index in range(num_examples):
                label = int(labels[index])
                if label > fold:
                    label -= 1
                elif label == fold:
                    label = max_digits
                try:
                    writer_seen.close()
                    writer_unseen.close()
                except:
                    pass
                writer_seen = tf.python_io.TFRecordWriter(os.path.join(path,  file+'_'+fname +'.tfrecord'))
                writer_unseen = tf.python_io.TFRecordWriter(os.path.join(path, file + '_heldout_' +  fname + '.tfrecord'))

                image_raw = images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tfrecord_utils.int64_feature([32]),
                    'width': tfrecord_utils.int64_feature([32]),
                    'depth': tfrecord_utils.int64_feature([3]),
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


def make_FA(args):
    path = os.path.join(args.target_path, "whole")
    os.makedirs(path)
    files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    data_X = None
    for file in files:
        _, data_X_new = unpickle(os.path.join(args.data_path, file))
        if data_X is not None:
            data_X = np.concatenate((data_X, data_X_new), axis=0)
        else:
            data_X=data_X_new

    collage = np.zeros(shape=[4 * 32, 8 * 32, 3])
    for i in xrange(32):
        j = i % 8
        k = i / 8
        img = data_X[i]
        collage[k * 32:(k + 1) * 32, j * 32:(j + 1) * 32, :] = img
    fig=plt.imshow(np.asarray(collage, dtype=np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(path, 'cifar_samples.png'), bbox_inches='tight')
    plt.close()

    components=100
    data_X = (np.reshape(data_X,(50000,3072))-127)/128.0
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


    np.savetxt(os.path.join(path, "fa_mean.txt"), mean)
    np.savetxt(os.path.join(path, "fa_variance.txt"), variance)
    np.savetxt(os.path.join(path, "fa_loading_matrix.txt"), load_matrix)

    samples = 32
    noise = 1
    z = np.random.multivariate_normal(np.zeros(shape=[components]), noise * np.eye(components), samples)
    x = np.random.multivariate_normal(np.zeros(shape=[3072]), np.eye(3072), samples)

    mu = np.dot(z, load_matrix) + mean
    images = (mu + np.sqrt(noise*variance)*x)*128+127
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
    plt.close()




def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_process_cifar_data.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if not os.path.isdir(args.target_path):
        os.makedirs(args.target_path)

    if args.makeFA == True:
        make_FA(args=args)
    if args.XVAL == True:
        process_data_XVAL(args)
    elif args.greyscale == True:
        process_data_greyscale(args)
    else:
        process_data(args)

if __name__ == '__main__':
    main()