import os
import sys
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
try:
    import cPickle as pickle
except:
    import pickle


class BaseModel(object):
    def __init__(self, network_architecture, name=None, save_path=None, load_path=None, debug_mode=0, seed=100):

        # Misc book-keeping parameters
        self._debug_mode = debug_mode
        self._save_path = save_path
        self._name = name

        # Specify default network. Generic enough for any model.
        self.network_architecture = network_architecture

        # Add check that network architecture has all the necessary parameters???
        # If new model, need to update architecture
        if load_path is None:
            assert network_architecture is not None
        else:
            # Load data (deserialize) architecture from path
            arch_path = os.path.join(load_path, 'model/net_arch.pickle')
            with open(arch_path, 'rb') as handle:
                self.network_architecture = pickle.load(handle)

        if (os.path.isfile(os.path.join(self._save_path, 'LOG.txt')) or os.path.isfile(
                os.path.join(self._save_path, 'model/weights.ckpt')) or os.path.isfile(
                os.path.join(self._save_path, 'model/net_arch.pickle'))) and load_path is None:
            print 'Model exists in directory - exiting.'
            sys.exit()
        if load_path is None:
            with open(os.path.join(self._save_path, 'LOG.txt'), 'w') as f:
                f.write('Creating Grader Model with configuration:\n')
                f.write('----------------------------------------------------------\n')
                for key in sorted(self.network_architecture.keys()):
                    f.write(key + ': ' + str(self.network_architecture[key]) + '\n')
                f.write('----------------------------------------------------------\n')

        # Parameters for training
        self._seed = seed
        self.initializer = self.network_architecture['initializer']

        # Tensorflow graph bookeeping
        self._graph = tf.Graph()
        # Construct Graph
        with self._graph.as_default():
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            tf.set_random_seed(self._seed)
            np.random.seed(self._seed)
            random.seed(self._seed)
            self.sess = tf.Session(config=config)

    def save(self, step=None):
        """ Saves model and parameters to self._save_path """
        with self._graph.as_default():
            path = os.path.join(self._save_path, 'model/weights.ckpt')
            if step is not None:
                self._saver.save(self.sess, path, global_step=step)
            else:
                self._saver.save(self.sess, path)
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                f.write('Saving Model to: ' + path + '\n')

        # Pickle network architecture into a file.
        path = os.path.join(self._save_path, 'model/net_arch.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(self.network_architecture, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, load_path, step=None):
        with self._graph.as_default():
            # If necessary, restore model from previous
            print 'loading model...'
            weights_path = 'model/weights.ckpt'
            if step is not None:
                weights_path = 'model/weights.ckpt-' + str(step)

            path = os.path.join(load_path, 'model/net_arch.pickle')
            with open(path, 'rb') as handle:
                self.network_architecture = pickle.load(handle)

            path = os.path.join(load_path, weights_path)
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                f.write('Restoring Model paratemters from: ' + path + '\n')
            self._saver.restore(self.sess, path)

    def _load_variables(self, load_scope, new_scope, load_path, trainable=False):
        # Restore parameters to DDN we are sampling from...
        if trainable:
            model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*" + new_scope + ".*")
        else:
            model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, ".*" + new_scope + ".*")
        dict = {}
        for model_var in model_variables:
            #print model_var.op.name, model_var.op.name.replace(new_scope, load_scope)
            dict[model_var.op.name.replace(new_scope, load_scope)] = model_var
        sampling_saver = tf.train.Saver(dict)
        param_path = os.path.join(load_path, 'model/weights.ckpt')
        sampling_saver.restore(self.sess, param_path)


    def _construct_dataset_from_tfrecord(self,
                                         filenames,
                                         _parse_func,
                                         _map_func,
                                         _batch_func,
                                         batch_size,
                                         capacity_mul=1000,
                                         num_threads=4,
                                         augment=False,
                                         train=False):
        with tf.device('/cpu:0'):
            capacity = capacity_mul*batch_size
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(_parse_func, num_parallel_calls=num_threads).cache()

            # Apply any other possible mapping possible mapping.
            dataset = _map_func(dataset, num_threads, capacity, augment=augment)

            # Apply shuffle dataset and repeat it indefinitely
            if train: dataset = dataset.shuffle(capacity, self._seed).repeat()

            # Apply shuffle dataset and repeat it indefinitely
            dataset = _batch_func(dataset, batch_size)

            if train:
                # Create an iterator for the dataset
                iterator = dataset.make_one_shot_iterator()
                return iterator.get_next(name='input_data')
            else:
                iterator = dataset.make_initializable_iterator()
                return iterator
