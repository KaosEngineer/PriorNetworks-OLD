#!/usr/bin/python
import argparse
import os
import sys

import matplotlib
import random
matplotlib.use('agg')

from PIL import Image
import tensorflow as tf
from sklearn.decomposition import FactorAnalysis as FA
from scipy.misc import imresize
from  core.utilities import tfrecord_utils

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
    sets=['train', 'val', 'test']
    index={}
    with open(os.path.join(args.data_path, 'index.txt'), 'r') as f:
        ind_list= [line[:-1].split() for line in f.readlines()]
    for i in xrange(len(ind_list)):
        index[ind_list[i][0]] = {'class' : i, 'name' : ind_list[i][1]}

    for set in sets:
        print set
        files = os.listdir(os.path.join(args.data_path,set+'/images/'))
        num_examples=len(files)
        #Make sure there is a good global shuffle
        random.shuffle(files)
        if set == 'val':
            with open(os.path.join(args.data_path, 'val/val_index.txt'), 'r') as f:
                val_list = [line[:-1].split() for line in f.readlines()]
            val_index={}
            for item in val_list:
                val_index[item[0]]= item[1]
        for file, i in zip(files, range(num_examples)):
            file_path = os.path.join(args.data_path, set+'/images/'+file)
            if set == 'train':
                label = index[file.split('_')[0]]['class']
                name = index[file.split('_')[0]]['name']
            elif set == 'val':
                code = val_index[file]
                label = index[code]['class']
                name = index[code]['name']
            else:
                label = -1
                name = 'NA'

            if i % 10000 == 0:
                print i
                try:
                    writer.close()
                except:
                    pass
                writer = tf.python_io.TFRecordWriter(os.path.join(args.target_path, set + '_' + str(i/10000) + '.tfrecord'))
            if os.path.isfile(file_path) and os.stat(file_path).st_size != 0:
                im = Image.open(file_path)
                im = im.resize((args.size, args.size), resample=Image.BICUBIC)
                image_raw = im.convert("RGB").tostring("raw", "RGB")

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tfrecord_utils.int64_feature([args.size]),
                    'width': tfrecord_utils.int64_feature([args.size]),
                    'depth': tfrecord_utils.int64_feature([3]),
                    'label': tfrecord_utils.int64_feature([label]),
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
    with open('CMDs/step_process_tinyImagenet_data.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    os.makedirs(args.target_path)
    process_data(args)


if __name__ == '__main__':
    main()