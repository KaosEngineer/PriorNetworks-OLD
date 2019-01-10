import os, sys
import time

import numpy as np
import tensorflow as tf
from scipy.stats import dirichlet
import core.utilities.utilities as util
from core.basemodel import BaseModel
from core.utilities import tfrecord_utils
from matplotlib import pyplot as plt
from matplotlib import cm

import tensorflow.contrib.slim as slim

try:
    import cPickle as pickle
except:
    import pickle
from scipy.stats import norm


def lrelu(x):
    return tf.maximum(x, 0.2 * x)

def clip_eta(eta, ord, eps):
    """
    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param ord: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epilson, bound of the perturbation.
    """

    # Clipping perturbation eta to self.ord norm ball
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')
    reduc_ind = list(xrange(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if ord == 1:
            norm = tf.maximum(avoid_zero_div,
                              tf.reduce_sum(tf.abs(eta),
                                            reduc_ind, keep_dims=True))
        elif ord == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero
            # in the gradient through this operation
            norm = tf.sqrt(tf.maximum(avoid_zero_div,
                                      tf.reduce_sum(tf.square(eta),
                                                    reduc_ind,
                                                    keep_dims=True)))
        # We must *clip* to within the norm ball, not *normalize* onto the
        # surface of the ball
        factor = tf.minimum(1., eps / norm)
        eta = eta * factor

    return eta

def permute_logits_least_likely(logits, batch_size):
    #logits = tf.Print(logits, [logits], message='original logits')
    max_values = tf.reduce_max(logits, axis=1)
    min_values = tf.reduce_min(logits, axis=1)

    max_inds = tf.expand_dims(tf.argmax(logits, axis=1), 1)
    min_inds = tf.expand_dims(tf.argmin(logits, axis=1), 1)

    values = tf.concat([max_values, min_values], axis=0)
    inds = tf.concat([min_inds, max_inds], axis=0)
    batch_inds = tf.cast(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [2, 1]),dtype=tf.int64)
    update_inds = tf.concat([batch_inds, inds], axis=1)
    update = tf.scatter_nd(update_inds, values, shape=[batch_size,10])
    #update = tf.Print(update, [update], message='Permute update')


    inds = tf.cast(tf.expand_dims(tf.range(10), 0),dtype=tf.int64)
    bb = tf.tile(max_inds, [1, 10])
    cc = tf.tile(min_inds, [1, 10])

    cond = tf.logical_or(tf.equal(inds, bb), tf.equal(inds, cc))
    permuted_logits = tf.stop_gradient(tf.where(cond, update, logits))

    assert_op=tf.assert_equal(tf.reduce_sum(permuted_logits,axis=1),
                              tf.reduce_sum(logits,axis=1))

    #with tf.control_dependencies([assert_op]):
   #     permuted_logits = tf.identity(permuted_logits)

    #permuted_logits = tf.Print(permuted_logits, [permuted_logits], message='permuted logits')
    return permuted_logits

def permute_logits_two_best(logits, batch_size):
    #logits = tf.Print(logits, [logits], message='original logits')
    values, indices = tf.nn.top_k(logits, k=2, sorted=True)
    print values.get_shape(), indices.get_shape()
    indices = tf.cast(indices,dtype=tf.int64)
    max_values = tf.slice(values, begin=[0,0], size=[-1,1])
    min_values = tf.slice(values, begin=[0,1], size=[-1,1])

    max_inds = tf.slice(indices, begin=[0,0], size=[-1,1])
    min_inds = tf.slice(indices, begin=[0,1], size=[-1,1])

    print max_values.get_shape(),min_values.get_shape()
    values = tf.squeeze(tf.concat([max_values, min_values], axis=0))

    inds = tf.concat([min_inds, max_inds], axis=0)
    batch_inds = tf.cast(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [2, 1]),dtype=tf.int64)
    update_inds = tf.concat([batch_inds, inds], axis=1)
    update = tf.scatter_nd(update_inds, values, shape=[batch_size,10])
    #update = tf.Print(update, [update], message='Permute update')

    inds = tf.cast(tf.expand_dims(tf.range(10), 0),dtype=tf.int64)
    bb = tf.tile(max_inds, [1, 10])
    cc = tf.tile(min_inds, [1, 10])

    cond = tf.logical_or(tf.equal(inds, bb), tf.equal(inds, cc))
    permuted_logits = tf.stop_gradient(tf.where(cond, update, logits))

    #assert_op=tf.assert_equal(tf.reduce_sum(permuted_logits,axis=1),
    #                          tf.reduce_sum(logits,axis=1))

    #with tf.control_dependencies([assert_op]):
    #    permuted_logits = tf.identity(permuted_logits)

    #permuted_logits = tf.Print(permuted_logits, [permuted_logits], message='permuted logits')
    return permuted_logits

def kl_divergence_pmf(target_probs, probs):
    return tf.reduce_sum(target_probs * (tf.log(target_probs) - tf.log(probs)), axis=1)

def kl_divergence_dirichlet_logits(target_alphas, alphas):
    target_alpha_0 = tf.reduce_sum(target_alphas, axis=1, keep_dims=True)
    alpha_0 = tf.reduce_sum(alphas, axis=1, keep_dims=True)

    return tf.lgamma(target_alpha_0) - tf.lgamma(alpha_0) \
           + tf.reduce_sum(tf.lgamma(alphas) - tf.lgamma(target_alphas)
                           , axis=1) \
           + tf.reduce_sum((target_alphas - alphas) * (tf.digamma(target_alphas) - tf.digamma(target_alpha_0))
                           , axis=1)

class PriorNet(BaseModel):
    def __init__(self, fa_path, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0,
                 seed=100, epoch=None):

        BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, save_path=save_path,
                           load_path=load_path, debug_mode=debug_mode)

        with self._graph.as_default():
            with tf.variable_scope('input') as scope:
                size = self.network_architecture['n_in']
                channels = self.network_architecture['n_channels']
                self._input_scope = scope
                self.batch_size = tf.placeholder(tf.int32, [])
                self.dropout = tf.placeholder(tf.float32, [])
                self.gnoise = tf.placeholder(tf.float32, [])
                self.images = tf.placeholder(tf.float32,
                                             [None, size, size, channels])  # Can I specify this automatically??
                # self.images = tf.placeholder(tf.float32, [None, size])  # Can I specify this automatically??

            with tf.variable_scope('PriorNet') as scope:
                # Not sure if this is even really necessary....
                if fa_path is not None:
                    fa_mean = np.loadtxt(os.path.join(fa_path, 'fa_mean.txt'), dtype=np.float32)
                    fa_variance = np.loadtxt(os.path.join(fa_path, 'fa_variance.txt'), dtype=np.float32)
                    fa_loading_matrix = np.loadtxt(os.path.join(fa_path, 'fa_loading_matrix.txt'), dtype=np.float32)
                    self.fa_mean = tf.expand_dims(tf.Variable(fa_mean, trainable=False), axis=0)
                    self.fa_variance = tf.Variable(fa_variance, trainable=False)
                    self.fa_loading_matrix = tf.Variable(fa_loading_matrix, trainable=False)

                # self.gmm_X, self.gmm_Y, self.gmm_noise = self._sample_gmm(batch_size=self.batch_size, noise=self.gnoise)

                self._model_scope = scope
                if fa_path is not None:
                    self.fa_images = self._sample_factor_analysis(batch_size=self.batch_size)
                self.mean, self.precision, self.logits = self._construct_network(x=self.images, keep_prob=self.dropout,
                                                                                 gain=self.network_architecture['gain'],
                                                                                 is_training=False)

            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if load_path is None:
            with self._graph.as_default():
                init = tf.global_variables_initializer()
                self.sess.run(init)

                # If necessary, restore model from previous
        elif load_path is not None:
            self.load(load_path=load_path, step=epoch)

    def parse_func(self, example_proto):

        features = tf.parse_single_example(
            serialized=example_proto,
            features={'height': tf.FixedLenFeature([], tf.int64),
                      'width': tf.FixedLenFeature([], tf.int64),
                      'depth': tf.FixedLenFeature([], tf.int64),
                      'label': tf.FixedLenFeature([], tf.int64),
                      'image_raw': tf.FixedLenFeature([], tf.string)})

        return features['label'], features['image_raw']

    def ts_parse_func(self, example_proto):
        n_out = self.network_architecture['n_out']
        features = tf.parse_single_example(
            serialized=example_proto,
            features={'height': tf.FixedLenFeature([], tf.int64),
                      'width': tf.FixedLenFeature([], tf.int64),
                      'depth': tf.FixedLenFeature([], tf.int64),
                      'label': tf.FixedLenFeature([], tf.int64),
                      'probs': tf.FixedLenFeature([n_out], tf.float32),
                      'image_raw': tf.FixedLenFeature([], tf.string)})

        return features['label'], features['probs'], features['image_raw']

    def map_func(self, dataset, num_threads, capacity, augment=False):

        dataset = dataset.map(lambda lab, img: (lab, tf.decode_raw(img, tf.uint8)),
                              num_parallel_calls=num_threads).prefetch(capacity)

        dim = self.network_architecture['n_in']
        channels = self.network_architecture['n_channels']
        dataset = dataset.map(lambda lab, img: (lab, tf.reshape(img, [dim, dim, channels])),
                              num_parallel_calls=num_threads).prefetch(capacity)

        if augment is True:
            dataset = dataset.map(lambda lab, img: (lab, tf.image.random_flip_left_right(img)),
                                  num_parallel_calls=num_threads).prefetch(capacity)
            # dataset = dataset.map(lambda lab, img: (lab, tf.image.random_flip_up_down(img)),
            #                      num_parallel_calls=num_threads).prefetch(capacity)
            dataset = dataset.map(
                lambda lab, img: (lab, tf.pad(img, paddings=[[4, 4], [4, 4], [0, 0]], mode='REFLECT')),
                num_parallel_calls=num_threads).prefetch(capacity)
            dataset = dataset.map(lambda lab, img: (lab, tf.random_crop(img, size=[dim, dim, channels])),
                                  num_parallel_calls=num_threads).prefetch(capacity)
            dataset = dataset.map(
                lambda lab, img: (lab, tf.contrib.image.rotate(img, tf.random_uniform([], minval=-0.25, maxval=0.25))),
                num_parallel_calls=num_threads).prefetch(capacity)

        # dataset = dataset.map(lambda lab, img : (lab, tf.transpose(img, [2, 0, 1])),
        #                       num_threads=num_threads).prefetch(capacity)

        dataset = dataset.map(lambda lab, img: (lab, (tf.cast(img, dtype=tf.float32) - 127.0) / 128.0),
                              num_parallel_calls=num_threads).prefetch(capacity)

        return dataset

    def ts_map_func(self, dataset, num_threads, capacity, augment=False):

        dataset = dataset.map(lambda lab, probs, img: (lab, probs, tf.decode_raw(img, tf.uint8)),
                              num_parallel_calls=num_threads).prefetch(capacity)

        dim = self.network_architecture['n_in']
        channels = self.network_architecture['n_channels']
        dataset = dataset.map(lambda lab, probs, img: (lab, probs, tf.reshape(img, [dim, dim, channels])),
                              num_parallel_calls=num_threads).prefetch(capacity)

        if augment is True:
            dataset = dataset.map(lambda lab, probs, img: (lab, probs, tf.image.random_flip_left_right(img)),
                                  num_parallel_calls=num_threads).prefetch(capacity)
            # dataset = dataset.map(lambda lab, img: (lab, tf.image.random_flip_up_down(img)),
            #                      num_parallel_calls=num_threads).prefetch(capacity)
            dataset = dataset.map(
                lambda lab, probs, img: (lab, probs, tf.pad(img, paddings=[[4, 4], [4, 4], [0, 0]], mode='REFLECT')),
                num_parallel_calls=num_threads).prefetch(capacity)
            dataset = dataset.map(lambda lab, probs, img: (lab, probs, tf.random_crop(img, size=[dim, dim, channels])),
                                  num_parallel_calls=num_threads).prefetch(capacity)
            dataset = dataset.map(
                lambda lab, probs, img: (lab, probs, tf.contrib.image.rotate(img, tf.random_uniform([], minval=-0.25, maxval=0.25))),
                num_parallel_calls=num_threads).prefetch(capacity)

        # dataset = dataset.map(lambda lab, img : (lab, tf.transpose(img, [2, 0, 1])),
        #                       num_threads=num_threads).prefetch(capacity)

        dataset = dataset.map(lambda lab,probs, img: (lab, probs, (tf.cast(img, dtype=tf.float32) - 127.0) / 128.0),
                              num_parallel_calls=num_threads).prefetch(capacity)

        return dataset

    def noise_map_func(self, dataset, num_threads, capacity, augment=None):

        dataset = dataset.map(lambda lab, img: (lab, tf.decode_raw(img, tf.uint8)),
                              num_parallel_calls=num_threads).prefetch(capacity)

        dim = self.network_architecture['n_in']
        channels = self.network_architecture['n_channels']
        dataset = dataset.map(lambda lab, img: (lab, tf.reshape(img, [dim, dim, channels])),
                              num_parallel_calls=num_threads).prefetch(capacity)

        dataset = dataset.map(lambda lab, img: (lab, tf.image.random_flip_left_right(img)),
                              num_parallel_calls=num_threads).prefetch(capacity)
        dataset = dataset.map(lambda lab, img: (lab, tf.image.random_flip_up_down(img)),
                              num_parallel_calls=num_threads).prefetch(capacity)
        dataset = dataset.map(lambda lab, img: (lab, tf.pad(img, paddings=[[4, 4], [4, 4], [0, 0]], mode='REFLECT')),
                              num_parallel_calls=num_threads).prefetch(capacity)
        dataset = dataset.map(lambda lab, img: (lab, tf.random_crop(img, size=[32, 32, 3])),
                              num_parallel_calls=num_threads).prefetch(capacity)
        dataset = dataset.map(lambda lab, img: (lab, tf.contrib.image.rotate(img,
                                                                             tf.random_uniform([], minval=-0.25,
                                                                                               maxval=0.25))),
                              num_parallel_calls=num_threads).prefetch(capacity)

        # dataset = dataset.map(lambda lab, img: (lab, tf.image.random_hue(img,0.01,seed=self._seed)),
        #                       num_parallel_calls=num_threads).prefetch(capacity)
        # dataset = dataset.map(lambda lab, img: (lab, tf.image.random_saturation(img,0.95, 1.05,self._seed)),
        #                       num_parallel_calls=num_threads).prefetch(capacity)
        # dataset = dataset.map(lambda lab, img: (lab, tf.image.random_contrast(img,0.95, 1.05,self._seed)),
        #                       num_parallel_calls=num_threads).prefetch(capacity)

        dataset = dataset.map(lambda lab, img: (lab, (tf.cast(img, dtype=tf.float32) - 127.0) / 128.0),
                              num_parallel_calls=num_threads).prefetch(capacity)

        return dataset

    def batch_func(self, dataset, batch_size):

        batched_dataset = dataset.batch(batch_size=batch_size)

        return batched_dataset

    def _sample_gmm(self, batch_size, noise, scale=4.0):

        init_weights = [1 / 3.0, 1 / 3.0, 1 / 3.0]

        # generate spherical data centered on (20, 20)
        mu_1 = scale * np.asarray([0, 1.0], dtype=np.float32)
        mu_2 = scale * np.asarray([-np.sqrt(3) / 2, -1.0 / 2], dtype=np.float32)
        mu_3 = scale * np.asarray([np.sqrt(3) / 2, -1.0 / 2], dtype=np.float32)
        diag_stdev = tf.ones([2]) * noise  # np.asarray([noise, noise], dtype=np.float32)
        gaussian1 = tf.contrib.distributions.MultivariateNormalDiag(mu_1, diag_stdev)
        gaussian2 = tf.contrib.distributions.MultivariateNormalDiag(mu_2, diag_stdev)
        gaussian3 = tf.contrib.distributions.MultivariateNormalDiag(mu_3, diag_stdev)
        categorical = tf.contrib.distributions.Categorical(init_weights)
        gmm = tf.contrib.distributions.Mixture(cat=categorical, components=[gaussian1, gaussian2, gaussian3])

        # Generate Samples for each of the classes
        samples_1 = gaussian1.sample(batch_size / 3)
        samples_2 = gaussian2.sample(batch_size / 3)
        samples_3 = gaussian3.sample(batch_size / 3)
        samples_x = tf.concat([samples_1, samples_2, samples_3], axis=0)
        samples_y = tf.concat([0 * np.ones(shape=[150 / 3]), np.ones(shape=[150 / 3]), 2 * np.ones(shape=[150 / 3])],
                              axis=0)
        thresh = (norm.pdf(3.1)) ** 2
        noise = tf.random_uniform(shape=[3 * 150, 2], minval=-20, maxval=20.0, dtype=tf.float32, seed=self._seed)
        samples_y = tf.cast(samples_y, dtype=tf.int64)

        probs = gmm.prob(noise)
        noise = tf.squeeze(tf.gather(noise, tf.where(tf.less_equal(probs, thresh)), axis=0), axis=1)
        noise = tf.slice(noise, [0, 0], size=[150, -1])
        print noise.get_shape()

        return samples_x, samples_y, noise

    def _sample_factor_analysis(self, batch_size, epsilon=1e-6):
        n_z = self.network_architecture['n_z']
        dim = self.network_architecture['n_in']
        channels = self.network_architecture['n_channels']
        size = dim * dim * channels
        # noise = tf.truncated_normal(shape=[1], mean=0.0, stddev=5.0, dtype=tf.float32, seed=self._seed)
        noise = tf.random_uniform(shape=[1], minval=0.01, maxval=5.0, dtype=tf.float32, seed=self._seed)
        z = tf.sqrt(1e-6 + noise) * tf.random_normal((batch_size, n_z), 0.0, 1.0, dtype=tf.float32, seed=self._seed)
        eta = tf.random_normal((batch_size, size), 0.0, 1.0, dtype=tf.float32, seed=self._seed)
        images = tf.matmul(z, self.fa_loading_matrix) + self.fa_mean + tf.sqrt(epsilon + noise * self.fa_variance) * eta
        # images = tf.Print(images, [images], message="my Z-values:",
        #                  summarize=784)  # <-------- TF PRINT STATMENT

        channels = self.network_architecture['n_channels']
        images = tf.reshape(images, shape=[batch_size, dim, dim, channels])

        return tf.nn.tanh(images)

    def _construct_cost(self, labels, logits, is_training=False):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def _sample_z(self, batch_size, sigma=3.0, r=5.0):
        n_z = self.network_architecture['n_z']
        z = tf.random_uniform((25 * batch_size, n_z), -r, r, dtype=tf.float64, seed=self._seed)
        mu = tf.zeros(shape=[n_z], dtype=tf.float64)
        cov = sigma * tf.diag(tf.ones(shape=[n_z], dtype=tf.float64))
        MVN = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
        probs = MVN.prob(z)
        rv = norm()
        uni_prob = tf.ones_like(probs) * (rv.pdf(sigma)) ** (n_z)
        z = tf.squeeze(tf.gather(z, tf.where(tf.greater(uni_prob, probs))), axis=1)
        z = tf.cast(z, dtype=tf.float32)
        # z=tf.slice(z, begin=tf.zeros(shape=[n_z], dtype=tf.int32), size=[batch_size])
        return z[:batch_size, :]

    def _construct_KL_cost(self, target_mean, mean, target_precision, precision, epsilon=1e-8, is_training=False):
        cost = tf.lgamma(target_precision + epsilon) - tf.lgamma(precision + epsilon) \
               + tf.reduce_sum(
            tf.lgamma(mean * precision + epsilon) - tf.lgamma(target_mean * target_precision + epsilon), axis=1) \
               + tf.reduce_sum((target_precision * target_mean - mean * precision) * (
        tf.digamma(target_mean * target_precision + epsilon) -
        tf.digamma(target_precision + epsilon)), axis=1)
        cost = tf.reduce_mean(cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def _construct_MI_cost(self, logits, epsilon=1e-8):
        alphas = tf.exp(logits) + epsilon
        alpha_0 = tf.reduce_sum(alphas, axis=1, keep_dims=True)
        MI = -tf.reduce_sum(
            alphas / alpha_0 * (tf.log(alphas) - tf.log(alpha_0) - tf.digamma(alphas + 1) + tf.digamma(alpha_0 + 1)),
            axis=1)
        MI = tf.reduce_mean(MI)
        return MI

    def _add_bernoulli_noise(self, input, prob):
        bernoulli = tf.distributions.Bernoulli(probs=prob)
        noise = tf.cast(bernoulli.sample(sample_shape=tf.shape(input), seed=self._seed), dtype=tf.float32)
        return input + noise

    def _add_gaussian_noise(self, input, std):
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise

    def _construct_DE_cost(self, logits, epsilon=1e-8):
        n_classes = self.network_architecture['n_out']
        alphas = tf.exp(logits) + epsilon
        alpha_0 = tf.reduce_sum(alphas, axis=1)
        dentropy = tf.reduce_sum(tf.lgamma(alphas), axis=1) - tf.lgamma(alpha_0) + (alpha_0 - n_classes) * tf.digamma(
            alpha_0) \
                   - tf.reduce_sum((alphas - 1.0) * tf.digamma(alphas), axis=1)
        print dentropy.get_shape()
        return tf.reduce_mean(dentropy)

    def _construct_NLL_cost(self, labels, logits, smoothing=1e-2, epsilon=1e-8, is_training=False):
        alphas = tf.exp(logits) + epsilon
        n_classes = self.network_architecture['n_out']
        targets = tf.one_hot(labels, n_classes, dtype=tf.float32) * (1.0 - float(n_classes) * smoothing)
        print targets.get_shape()
        smooth = smoothing * tf.ones_like(targets)
        targets = targets + smooth
        alpha_0 = tf.reduce_sum(alphas, axis=1)

        print alphas.get_shape(), alpha_0.get_shape(), targets.get_shape()
        loss = tf.lgamma(alpha_0) - tf.reduce_sum(tf.lgamma(alphas), axis=1) + tf.reduce_sum(
            (alphas - 1.0) * tf.log(targets + smoothing), axis=1)
        cost = -tf.reduce_mean(loss)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def _FGSM_adversarial_example(self, eval_labels, eval_logits, eval_images, noise=None, anti_detect=0.0):
        if noise is None:
            noise = tf.abs(tf.random_normal(shape=[], mean=0, stddev=0.5, dtype=tf.float32, seed=self._seed)) + 0.0079
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_labels, logits=eval_logits)
        if anti_detect > 0.0:
            n_classes = self.network_architecture['n_out']
            alphas = tf.exp(eval_logits) + 1e-8
            alpha_0 = tf.reduce_sum(alphas, axis=1)
            dentropy = tf.reduce_sum(tf.lgamma(alphas), axis=1) - tf.lgamma(alpha_0) + (
                                                                                       alpha_0 - n_classes) * tf.digamma(
                alpha_0) - tf.reduce_sum((alphas - 1.0) * tf.digamma(alphas), axis=1)
            loss += anti_detect * dentropy
        grads = tf.sign(tf.gradients(loss, eval_images)[0])
        adversarial_images = eval_images + grads * noise
        adversarial_images = tf.clip_by_value(adversarial_images, -127.0 / 128.0, 1.0)
        adversarial_images = tf.stop_gradient(adversarial_images)
        return adversarial_images

    # def FGM_attack(eval_labels, eval_logits, eval_images, eps, ord=np.inf, batch_size=None, seed=100):
    #
    #     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_labels, logits=eval_logits)
    #     grads = tf.gradients(loss, eval_images)[0]
    #     if ord == np.inf:
    #         grads = tf.stop_gradient(tf.sign(grads))
    #     elif ord == 1:
    #         grads = grads / tf.reduce_sum(tf.abs(grads), reduction_indices=[1, 2, 3], keep_dims=True)
    #     elif ord == 2:
    #         grads = grads / tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1, 2, 3], keep_dims=True))
    #     else:
    #         raise NotImplementedError("Only L-inf, L1 and L2 norms are "
    #                                   "currently implemented.")
    #     grads = tf.sign(tf.gradients(loss, eval_images)[0])
    #     adversarial_images = eval_images + grads * eps
    #     adversarial_images = tf.clip_by_value(adversarial_images, -127.0 / 128.0, 1.0)
    #     adversarial_images = tf.stop_gradient(adversarial_images)
    #     return adversarial_images

    def _targeted_FGSM_adversarial_example(self, eval_labels, eval_logits, eval_images, batch_size, noise=None,
                                           anti_detect=0.0):
        sampled_labels = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.network_architecture['n_out'] - 1,
                                           dtype=tf.int64)
        sign = tf.where(tf.equal(eval_labels, sampled_labels),
                        tf.ones(shape=[batch_size, 1, 1, 1], dtype=tf.float32),
                        -tf.ones(shape=[batch_size, 1, 1, 1], dtype=tf.float32))
        if noise is None:
            noise = tf.abs(tf.random_normal(shape=[], mean=0, stddev=0.5, dtype=tf.float32, seed=self._seed)) + 0.0079
        # targets = tf.arg_min(eval_probs, dimension=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sampled_labels, logits=eval_logits)
        if anti_detect > 0.0:
            n_classes = self.network_architecture['n_out']
            alphas = tf.exp(eval_logits) + 1e-8
            alpha_0 = tf.reduce_sum(alphas, axis=1)
            dentropy = tf.reduce_sum(tf.lgamma(alphas), axis=1) - tf.lgamma(alpha_0) + (
                                                                                       alpha_0 - n_classes) * tf.digamma(
                alpha_0) - tf.reduce_sum((alphas - 1.0) * tf.digamma(alphas), axis=1)
            loss += anti_detect * dentropy
        grads = tf.sign(tf.gradients(loss, eval_images)[0])
        adversarial_images = tf.stop_gradient(
            tf.maximum(tf.minimum(eval_images + sign * grads * noise, 1.0), -127.0 / 128.0))
        return adversarial_images

    def fit(self,
            train_pattern,
            valid_pattern,
            n_examples,
            augment,
            learning_rate=1e-2,
            cycle_length=30,
            lr_decay=1.0,
            dropout=1.0,
            batch_size=64,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=30):
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_batches = n_examples / batch_size

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                train_filenames = tf.gfile.Glob(train_pattern)
                labels, images = self._construct_dataset_from_tfrecord(filenames=train_filenames,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=10000,
                                                                       num_threads=8,
                                                                       augment=augment,
                                                                       train=True)
                valid_filenames = tf.gfile.Glob(valid_pattern)
                valid_iterator = self._construct_dataset_from_tfrecord(filenames=valid_filenames,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=100,
                                                                       num_threads=8,
                                                                       train=False)
                valid_labels, valid_images = valid_iterator.get_next(name='valid_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                # Construct Training model
                mean, precision, logits = self._construct_network(x=images, keep_prob=self.dropout,
                                                                  gain=self.network_architecture['gain'],
                                                                  is_training=True)
                # adversarial_images = self._FGSM_adversarial_example(labels, logits, images)
                # adv_mean, adv_precision, adv_logits = self._construct_network(x=adversarial_images,keep_prob=self.dropout,gain=self.network_architecture['gain'],is_training=True)
                # logits = tf.concat([logits, adv_logits], axis=0)
                # Construct Validation model
                valid_mean, precision, valid_logits = self._construct_network(x=valid_images, keep_prob=self.dropout,
                                                                              gain=self.network_architecture['gain'],
                                                                              is_training=False)

            # labels= tf.tile(labels, [2])
            cost, w_total = self._construct_cost(labels=labels, logits=logits, is_training=True)
            valid_cost = self._construct_cost(labels=valid_labels, logits=valid_logits, is_training=False)

            # Experiment with 1-cycle learning rates...
            half_cycle = cycle_length / 2
            boundaries = [i * n_batches for i in xrange(0, cycle_length)]
            boundaries.extend([i * n_batches for i in xrange(cycle_length, n_epochs)])

            values_lr = [(learning_rate / 10.0 + (learning_rate - learning_rate / 10.0) / half_cycle * float(i)) for i
                         in xrange(0, half_cycle)]
            values_lr.extend(
                [(learning_rate / 10.0 + (learning_rate - learning_rate / 10.0) * (half_cycle - float(i)) / half_cycle)
                 for i in xrange(0, half_cycle)])
            values_lr.extend(
                [learning_rate / 10.0 - (learning_rate / 10.0 - 1e-6) / (n_epochs - cycle_length) * i for i in
                 xrange(0, n_epochs - cycle_length)])

            learning_rate_pwc = tf.train.piecewise_constant(global_step, boundaries, values_lr, name='lr')
            train = slim.learning.create_train_op(total_loss=w_total,
                                                  optimizer=optimizer(learning_rate=learning_rate_pwc,
                                                                      **optimizer_params),
                                                  global_step=global_step,
                                                  clip_gradient_norm=10.0,
                                                  check_numerics=True,
                                                  gate_gradients=tf.train.Optimizer.GATE_OP,
                                                  colocate_gradients_with_ops=False,
                                                  summarize_gradients=False)

            # train = util.create_train_op(total_loss=w_total,
            #                                 learning_rate=learning_rate,
            #                                 optimizer=optimizer,
            #                                 optimizer_params=optimizer_params,
            #                                 n_examples=n_examples,
            #                                 batch_size=batch_size,
            #                                 learning_rate_decay=lr_decay,
            #                                 global_step=global_step,
            #                                 clip_gradient_norm=10.0,
            #                                 summarize_gradients=False)




            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nOptimizer: %s\nDropout: %f\n')
                f.write(
                    format_str % (learning_rate, lr_decay, batch_size, str(optimizer), dropout) + '\n\n')

            print "Beginning training..."
            format_str = (
            'Epoch %d, Train Loss = %.2f , Valid Loss = %.2f, Valid Accuracy = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
            start_time = time.time()
            for epoch in xrange(1, n_epochs + 1):
                batch_time = time.time()
                total_loss = 0.0
                for batch in xrange(n_batches):
                    _, loss = self.sess.run([train, cost],
                                            feed_dict={self.dropout: dropout,
                                                       self.batch_size: batch_size})
                    total_loss += loss
                total_loss /= n_batches

                duration = time.time() - batch_time
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                valid_loss = 0.0
                count = 0.0
                all_preds = None
                all_labels = None
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        count += 1.0
                        loss, labs, preds = self.sess.run([valid_cost, valid_labels, tf.argmax(valid_mean, axis=1)],
                                                          feed_dict={self.dropout: 1.0})
                        valid_loss += loss
                        if all_preds is None:
                            all_preds = preds[:, np.newaxis]
                            all_labels = labs[:, np.newaxis]
                        else:
                            all_labels = np.concatenate((all_labels, labs[:, np.newaxis]), axis=0)
                            all_preds = np.concatenate((all_preds, preds[:, np.newaxis]), axis=0)
                    except tf.errors.OutOfRangeError:
                        break
                accuracy = np.mean(np.asarray(all_labels == all_preds, dtype=np.float32)) * 100
                valid_loss /= count
                print (format_str % (epoch, total_loss, valid_loss, accuracy, examples_per_sec, sec_per_epoch))
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(
                        format_str % (epoch, total_loss, valid_loss, accuracy, examples_per_sec, sec_per_epoch) + '\n')
                self.save()

            duration = start_time - time.time()
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % duration + '\n')
                f.write('----------------------------------------------------------\n')
            print (format_str % duration)

    def fit_with_noise(self,
                       train_pattern,
                       valid_pattern,
                       n_examples,
                       alpha_target,
                       loss,
                       smoothing,
                       mode,
                       noise_pattern=None,
                       augment=False,
                       adv_targeted=False,
                       anti_detect=0.0,
                       learning_rate=1e-2,
                       cycle_length=30,
                       lr_decay=1.0,
                       dropout=1.0,
                       ce_weight=1.0,
                       noise_weight=1.0,
                       batch_size=64,
                       optimizer=tf.train.AdamOptimizer,
                       optimizer_params={},
                       n_epochs=30):
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_batches = n_examples / batch_size

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                train_filenames = tf.gfile.Glob(train_pattern)
                valid_filenames = tf.gfile.Glob(valid_pattern)

                labels, images = self._construct_dataset_from_tfrecord(filenames=train_filenames,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=10000,
                                                                       num_threads=8,
                                                                       augment=augment,
                                                                       train=True)
                valid_iterator = self._construct_dataset_from_tfrecord(filenames=valid_filenames,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=100,
                                                                       num_threads=8,
                                                                       train=False)
                valid_labels,  valid_images = valid_iterator.get_next(name='valid_data')
                if noise_pattern is not None:
                    noise_filenames = tf.gfile.Glob(noise_pattern)
                    _, noise_images = self._construct_dataset_from_tfrecord(filenames=noise_filenames,
                                                                            _parse_func=self.parse_func,
                                                                            _batch_func=self.batch_func,
                                                                            _map_func=self.noise_map_func,
                                                                            batch_size=batch_size,
                                                                            capacity_mul=10000,
                                                                            num_threads=8,
                                                                            augment=True,
                                                                            train=True)

            # Construct Training model
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                mean, precision, logits = self._construct_network(x=images, keep_prob=self.dropout,
                                                                  gain=self.network_architecture['gain'],
                                                                  is_training=True)
                if adv_targeted == True:
                    adversarial_images = self._targeted_FGSM_adversarial_example(labels, logits, images, batch_size,
                                                                                 anti_detect=anti_detect)
                else:
                    adversarial_images = self._FGSM_adversarial_example(labels, logits, images, anti_detect=anti_detect)
                valid_mean, valid_precision, valid_logits = self._construct_network(x=valid_images,
                                                                                    keep_prob=self.dropout,
                                                                                    gain=self.network_architecture[
                                                                                        'gain'], is_training=False)
                if noise_pattern is not None:
                    noise_images = noise_images
                else:
                    noise_images = self.fa_images
                noise_mean, noise_precision, noise_logits = self._construct_network(x=noise_images,
                                                                                    keep_prob=self.dropout,
                                                                                    gain=self.network_architecture[
                                                                                        'gain'], is_training=True)
                adv_mean, adv_precision, adv_logits = self._construct_network(x=adversarial_images,
                                                                              keep_prob=self.dropout,
                                                                              gain=self.network_architecture['gain'],
                                                                              is_training=True)

            # In-Domain Loss
            valid_CE_cost = self._construct_cost(labels=valid_labels, logits=valid_logits, is_training=False)
            xent_cost = self._construct_cost(labels=labels, logits=logits, is_training=False)
            smoothing = smoothing
            if loss == 'KL':
                n_classes = self.network_architecture['n_out']
                labels_one_hot = tf.one_hot(labels, n_classes, dtype=tf.float32) * (1.0 - float(n_classes) * smoothing)
                labels_one_hot += smoothing * tf.ones_like(labels_one_hot)
                target_precision_train = alpha_target * tf.ones([batch_size, 1], dtype=tf.float32)  # +noise
                cost, KL_cost = self._construct_KL_cost(labels_one_hot, mean, target_precision_train, precision,
                                                        is_training=True)

                # Validations Losses
                valid_labels_one_hot = tf.one_hot(valid_labels, n_classes, dtype=tf.float32) * (
                1.0 - float(n_classes) * smoothing)
                valid_labels_one_hot += smoothing * tf.ones_like(valid_labels_one_hot)
                target_precision_valid = alpha_target * tf.expand_dims(tf.ones_like(valid_labels, dtype=tf.float32),
                                                                       axis=1)
                valid_cost = self._construct_KL_cost(valid_labels_one_hot, valid_mean, target_precision_valid,
                                                     valid_precision, is_training=False)

                # Synthetic Noise Loss
                target_precision = float(n_classes) * tf.ones([batch_size, 1], dtype=tf.float32)  # +tf.abs(noise)
                target_mean = tf.ones([batch_size, n_classes], dtype=tf.float32) / float(n_classes)
                noise_cost = self._construct_KL_cost(target_mean, noise_mean, target_precision, noise_precision,
                                                     is_training=False)
                adv_cost = self._construct_KL_cost(mean, adv_mean, target_precision, adv_precision, is_training=False)

                if mode == 'OOD':
                    total_cost = xent_cost * ce_weight + KL_cost + noise_weight * noise_cost
                elif mode == 'ADV':
                    noise_cost = adv_cost
                    total_cost = xent_cost * ce_weight + KL_cost + noise_weight * noise_cost
                elif mode == 'ADV+OOD':
                    noise_cost += adv_cost
                    total_cost = xent_cost * ce_weight + KL_cost + noise_weight * noise_cost * 0.5
                elif mode == 'ID':
                    total_cost = xent_cost * ce_weight + KL_cost
                    noise_cost = tf.zeros([],dtype=tf.float32)

            elif loss == 'NLL-MI':
                cost, NLL_cost = self._construct_NLL_cost(labels, logits, smoothing=smoothing, is_training=True)
                valid_cost = self._construct_NLL_cost(valid_labels, valid_logits, smoothing=smoothing)
                noise_cost = self._construct_MI_cost(noise_logits)
                adv_cost = self._construct_MI_cost(adv_logits)
                total_cost = ce_weight * xent_cost + NLL_cost - noise_weight * noise_cost
            else:
                cost, NLL_cost = self._construct_NLL_cost(labels, logits, smoothing=smoothing, is_training=True)
                valid_cost = self._construct_NLL_cost(valid_labels, valid_logits, smoothing=smoothing)
                noise_cost = self._construct_DE_cost(noise_logits)
                adv_cost = self._construct_DE_cost(adv_logits)
                total_cost = ce_weight * xent_cost + NLL_cost - noise_weight * noise_cost

            model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "PriorNet.*")
            # train_op = util.create_train_op(total_loss=total_cost,
            #                                 learning_rate=learning_rate,
            #                                 optimizer=optimizer,
            #                                 optimizer_params=optimizer_params,
            #                                 n_examples=n_examples,
            #                                 batch_size=batch_size,
            #                                 variables_to_train=model_variables,
            #                                 learning_rate_decay=lr_decay,
            #                                 global_step=global_step,
            #                                 clip_gradient_norm=10.0,
            #                                 summarize_gradients=False)

            half_cycle = cycle_length / 2
            boundaries = [i * n_batches for i in xrange(0, cycle_length)]
            boundaries.extend([i * n_batches for i in xrange(cycle_length, n_epochs)])

            values_lr = [(learning_rate / 10.0 + (learning_rate - learning_rate / 10.0) / half_cycle * float(i)) for i
                         in xrange(0, half_cycle)]
            values_lr.extend(
                [(learning_rate / 10.0 + (learning_rate - learning_rate / 10.0) * (half_cycle - float(i)) / half_cycle)
                 for i in xrange(0, half_cycle)])
            values_lr.extend(
                [learning_rate / 10.0 - (learning_rate / 10.0 - 1e-6) / (n_epochs - cycle_length) * i for i in
                 xrange(0, n_epochs - cycle_length)])

            learning_rate_pwc = tf.train.piecewise_constant(global_step, boundaries, values_lr)
            train_op = slim.learning.create_train_op(total_loss=total_cost,
                                                     optimizer=optimizer(learning_rate=learning_rate_pwc,
                                                                         **optimizer_params),
                                                     global_step=global_step,
                                                     clip_gradient_norm=10.0,
                                                     variables_to_train=model_variables,
                                                     check_numerics=True,
                                                     gate_gradients=tf.train.Optimizer.GATE_OP,
                                                     colocate_gradients_with_ops=False,
                                                     summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nOptimizer: %s\nDropout: %f\n')
                f.write(
                    format_str % (learning_rate, lr_decay, batch_size, str(optimizer), dropout) + '\n\n')

            # merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./', self.sess.graph)
            print "Beginning training..."
            format_str = (
            'Epoch %d, KL_DLoss = %.2f, XE_DLoss = %.2f, KL_NLoss = %.3f, Valid Acc = %.2f, Valid KL_DLoss = %.2f, Valid XE_DLoss = %.2f, Valid KL_NLoss = %.3f, VP = %.3f, NP = %.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
            start_time = time.time()
            for epoch in xrange(1, n_epochs + 1):
                batch_time = time.time()
                total_KLD_loss = 0.0
                total_XED_loss = 0.0
                total_KLN_loss = 0.0
                for batch in xrange(n_batches):
                    _, l, KL_Nloss, KL_Dloss, XE_Dloss = self.sess.run(
                        [train_op, precision, noise_cost, cost, xent_cost],
                        feed_dict={self.dropout: dropout,
                                   self.batch_size: batch_size})
                    total_XED_loss += XE_Dloss
                    total_KLD_loss += KL_Dloss
                    total_KLN_loss += KL_Nloss
                    # train_writer.add_summary(summary, (epoch-1)*n_batches+batch)
                total_XED_loss /= n_batches
                total_KLD_loss /= n_batches
                total_KLN_loss /= n_batches

                duration = time.time() - batch_time
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                valid_XE_Dloss = 0.0
                valid_KL_Nloss = 0.0
                valid_KL_DLoss = 0.0
                mean_valid_prec = 0.0
                mean_noise_prec = 0.0
                count = 0.0
                all_preds = None
                all_labels = None
                print 'init valid iterator'
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        count += 1.0
                        labs, preds, XE_Dloss, KL_Dloss, KL_Nloss, vp, noisep = self.sess.run(
                            [valid_labels, tf.argmax(valid_mean, axis=1), valid_CE_cost, valid_cost, noise_cost,
                             valid_precision, noise_precision], feed_dict={self.dropout: 1.0,
                                                                           self.batch_size: batch_size})
                        mean_noise_prec += np.mean(noisep)
                        mean_valid_prec += np.mean(vp)
                        valid_KL_DLoss += KL_Dloss
                        valid_XE_Dloss += XE_Dloss
                        valid_KL_Nloss += KL_Nloss
                        if all_preds is None:
                            all_preds = preds[:, np.newaxis]
                            all_labels = labs[:, np.newaxis]
                        else:
                            all_labels = np.concatenate((all_labels, labs[:, np.newaxis]), axis=0)
                            all_preds = np.concatenate((all_preds, preds[:, np.newaxis]), axis=0)
                    except tf.errors.OutOfRangeError:
                        break
                accuracy = np.mean(np.asarray(all_labels == all_preds, dtype=np.float32)) * 100
                mean_noise_prec /= count
                mean_valid_prec /= count
                valid_XE_Dloss /= count
                valid_KL_Nloss /= count
                valid_KL_DLoss /= count
                print (format_str % (
                epoch, total_KLD_loss, total_XED_loss, total_KLN_loss, accuracy, valid_KL_DLoss, valid_XE_Dloss,
                valid_KL_Nloss, mean_valid_prec, mean_noise_prec, examples_per_sec, sec_per_epoch))
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (
                    epoch, total_KLD_loss, total_XED_loss, total_KLN_loss, accuracy, valid_KL_DLoss, valid_XE_Dloss,
                    valid_KL_Nloss, mean_valid_prec, mean_noise_prec, examples_per_sec, sec_per_epoch))
                self.save()

            duration = start_time - time.time()
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % duration + '\n')
                f.write('----------------------------------------------------------\n')
            print (format_str % duration)

    def fit_gmm(self,
                alpha_target,
                gnoise,
                learning_rate=1e-2,
                lr_decay=1.0,
                dropout=1.0,
                in_domain_weight=0.0,
                batch_size=64,
                optimizer=tf.train.AdamOptimizer,
                optimizer_params={},
                n_epochs=30):
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_examples = 30000
            n_batches = n_examples / batch_size

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            eval_data = tf.random_uniform(shape=[10000, 2], minval=-20, maxval=20.0, dtype=tf.float32, seed=self._seed)
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                mean, precision, logits = self._construct_network(x=self.gmm_X,
                                                                  keep_prob=self.dropout,
                                                                  gain=self.network_architecture['gain'],
                                                                  is_training=True)
                noise_mean, noise_precision, noise_logits = self._construct_network(x=self.gmm_noise,
                                                                                    keep_prob=self.dropout,
                                                                                    gain=self.network_architecture[
                                                                                        'gain'],
                                                                                    is_training=True)
                eval_mean, eval_prevision, eval_logits = self._construct_network(x=eval_data,
                                                                                 keep_prob=1.0,
                                                                                 gain=self.network_architecture['gain'],
                                                                                 is_training=False)
                eval_entropy = -tf.reduce_sum(eval_mean * tf.log(eval_mean), axis=1)
                eval_alphas = tf.exp(eval_logits)

            # In-Domain Loss
            n_classes = self.network_architecture['n_out']
            labels_one_hot = tf.one_hot(self.gmm_Y, n_classes, dtype=tf.float32) * (
            alpha_target - (float(n_classes))) / alpha_target + tf.ones([batch_size, n_classes],
                                                                        dtype=tf.float32) / alpha_target
            target_precision_train = alpha_target * tf.ones([batch_size, 1], dtype=tf.float32)
            cost, KL_cost = self._construct_KL_cost(labels_one_hot, mean, target_precision_train, precision,
                                                    is_training=True)
            xent_cost = self._construct_cost(labels=self.gmm_Y, logits=logits, is_training=False)

            # Synthetic Noise Loss
            target_precision = float(n_classes) * tf.ones([batch_size, 1], dtype=tf.float32)
            target_mean = tf.ones([batch_size, n_classes], dtype=tf.float32) / float(n_classes)
            fa_cost = self._construct_KL_cost(target_mean, noise_mean, target_precision, noise_precision,
                                              is_training=False)

            total_cost = in_domain_weight * xent_cost + KL_cost + fa_cost
            train_op = util.create_train_op(total_loss=total_cost,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nOptimizer: %s\nDropout: %f\n')
                f.write(
                    format_str % (learning_rate, lr_decay, batch_size, str(optimizer), dropout) + '\n\n')

            print "Beginning training..."
            format_str = (
            'Epoch %d, KL_DLoss = %.2f, XE_DLoss = %.2f, KL_NLoss = %.3f, Valid KL_DLoss = %.2f, Valid XE_DLoss = %.2f, Valid KL_NLoss = %.3f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            start_time = time.time()
            for epoch in xrange(1, n_epochs + 1):
                batch_time = time.time()
                total_KLD_loss = 0.0
                total_XED_loss = 0.0
                total_KLN_loss = 0.0
                for batch in xrange(n_batches):
                    _, KL_Nloss, KL_Dloss, XE_Dloss = self.sess.run([train_op, fa_cost, KL_cost, xent_cost],
                                                                    feed_dict={self.dropout: dropout,
                                                                               self.batch_size: batch_size,
                                                                               self.gnoise: gnoise})
                    total_XED_loss += XE_Dloss
                    total_KLD_loss += KL_Dloss
                    total_KLN_loss += KL_Nloss
                total_XED_loss /= n_batches
                total_KLD_loss /= n_batches
                total_KLN_loss /= n_batches

                duration = time.time() - batch_time
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                valid_XE_Dloss = 0.0
                valid_KL_Nloss = 0.0
                valid_KL_DLoss = 0.0
                print (format_str % (
                epoch, total_KLD_loss, total_XED_loss, total_KLN_loss, valid_KL_DLoss, valid_XE_Dloss, valid_KL_Nloss,
                examples_per_sec, sec_per_epoch))
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (
                    epoch, total_KLD_loss, total_XED_loss, total_KLN_loss, valid_KL_DLoss, valid_XE_Dloss,
                    valid_KL_Nloss, examples_per_sec, sec_per_epoch))
                self.save()

                entropy, alphas, inputs = self.sess.run([eval_entropy, eval_alphas, eval_data],
                                                        feed_dict={self.dropout: 1.0, self.batch_size: 10000})
                diff_entropy = np.asarray([dirichlet(alpha).entropy() for alpha in alphas])

                plt.scatter(inputs[:, 0], inputs[:, 1], c=entropy, cmap=cm.Blues, alpha=0.9)
                plt.xlim(-20, 20)
                plt.ylim(-20, 20)
                plt.savefig('Entropy.jpeg', bbox_inches='tight')
                plt.close()
                plt.scatter(inputs[:, 0], inputs[:, 1], c=diff_entropy, cmap=cm.Blues, alpha=0.9)
                plt.xlim(-20, 20)
                plt.ylim(-20, 20)
                plt.savefig('DEntropy.jpeg', bbox_inches='tight')
                plt.close()

            duration = start_time - time.time()
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % duration + '\n')
                f.write('----------------------------------------------------------\n')
            print (format_str % duration)

    def eval(self, eval_pattern, batch_size, n_samples, dropout, corruption=0.0):
        with self._graph.as_default():
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                eval_files = tf.gfile.Glob(eval_pattern)
                try:
                    eval_files.sort(key=lambda f: int(filter(str.isdigit, f) or -1))
                except:
                    pass
                valid_iterator = self._construct_dataset_from_tfrecord(filenames=eval_files,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=100,
                                                                       num_threads=8,
                                                                       train=False)
                eval_labels, eval_images = valid_iterator.get_next(name='eval_data')

                eval_images = tf.tile(eval_images, [n_samples, 1, 1, 1], name='replicated_images')
                eval_labels = tf.tile(eval_labels, [n_samples], name='replicated_labels')

            # noinspection PyInterpreter
            eval_images = self._add_gaussian_noise(eval_images, std=corruption)
            # Construct Validation model
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                eval_probs, eval_precision, eval_logits = self._construct_network(x=eval_images, keep_prob=self.dropout,
                                                                                  gain=self.network_architecture[
                                                                                      'gain'], is_training=False)

            count = 0
            self.sess.run(valid_iterator.initializer)
            all_labels = None
            all_probs = None
            all_logits = None
            while True:
                try:
                    labels, probs, logits = self.sess.run(
                        [eval_labels, eval_probs, eval_logits], feed_dict={self.dropout: dropout})
                    if count == 0:
                        all_labels, all_probs, all_logits = labels, probs, logits
                    else:
                        all_labels = np.concatenate((all_labels, labels), axis=0)
                        all_probs = np.concatenate((all_probs, probs), axis=0)
                        all_logits = np.concatenate((all_logits, logits), axis=0)
                    count += 1
                except tf.errors.OutOfRangeError:
                    break
            return all_labels, all_probs, all_logits

    def construct_teacher_tfrecord(self, eval_pattern, name, save_path, batch_size, examples_per_file):
        dim = self.network_architecture['n_in']
        channels = self.network_architecture['n_channels']
        with self._graph.as_default():
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                eval_files = tf.gfile.Glob(eval_pattern)
                valid_iterator = self._construct_dataset_from_tfrecord(filenames=eval_files,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=100,
                                                                       num_threads=8,
                                                                       train=False)
                eval_labels, eval_images = valid_iterator.get_next(name='eval_data')

            # Construct Validation model
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                eval_probs, _, _ = self._construct_network(x=eval_images, keep_prob=self.dropout,
                                                           gain=self.network_architecture['gain'], is_training=False)

            count = 0
            self.sess.run(valid_iterator.initializer)
            while True:
                try:
                    labels, probs, images = self.sess.run(
                        [eval_labels, eval_probs, eval_images], feed_dict={self.dropout: 1.0})
                    if count == 0:
                        all_labels, all_probs, all_images = labels, probs, images
                    else:
                        all_labels = np.concatenate((all_labels, labels), axis=0)
                        all_probs = np.concatenate((all_probs, probs), axis=0)
                        all_images = np.concatenate((all_images, images), axis=0)
                    count += 1
                except tf.errors.OutOfRangeError:
                    break

            all_images = np.asarray(all_images*128.0+127.0, dtype=np.uint8)
            for index in range(all_labels.shape[0]):
                label = int(all_labels[index])
                if index % examples_per_file == 0:
                    try:
                        writer.close()
                    except:
                        pass
                    writer = tf.python_io.TFRecordWriter(
                        os.path.join(save_path, name + '_teacher_' + str(index / examples_per_file) + '.tfrecord'))

                image_raw = all_images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tfrecord_utils.int64_feature([dim]),
                    'width': tfrecord_utils.int64_feature([dim]),
                    'depth': tfrecord_utils.int64_feature([channels]),
                    'label': tfrecord_utils.int64_feature([label]),
                    'probs': tfrecord_utils.float_feature(all_probs[index]),
                    'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
                writer.write(example.SerializeToString())
        try:
            writer.close()
        except:
            pass

    def compute_BIM_adversarial_examples(self, eval_pattern, batch_size, eps, alpha, iter, save_path, ord=np.inf,
                                         anti_detect=False, output_type='pmf', least_likely=False):
        with self._graph.as_default():
            with tf.variable_scope(self._input_scope, reuse=True) as scope:

                eval_files = tf.gfile.Glob(eval_pattern)
                valid_iterator = self._construct_dataset_from_tfrecord(filenames=eval_files,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=100,
                                                                       num_threads=1,
                                                                       train=False)
                eval_labels, eval_images = valid_iterator.get_next(name='eval_data')

            adv = eval_images
            eps = eps / 128.0

            if ord == 1:
                alpha = alpha * 1024.0 * 3.0
                eps = eps * 1024.0 * 3.0
            elif ord == 2:
                alpha = alpha * 32.0 * np.sqrt(3.0)
                eps = eps * 32.0 * np.sqrt(3.0)

            def cond(i, _, __, ___):
                return tf.less(i, iter)

            def body(i, adv, x, permuted_logits):

                with tf.variable_scope(self._model_scope, reuse=True) as scope:
                    eval_probs, eval_precision, eval_logits = self._construct_network(x=adv, keep_prob=self.dropout,
                                                                                      gain=self.network_architecture[
                                                                                          'gain'], is_training=False)

                if anti_detect:
                    if least_likely==True:
                        f1 = lambda: tf.stop_gradient(permute_logits_least_likely(eval_logits, batch_size))
                    else:
                        f1 = lambda: tf.stop_gradient(permute_logits_two_best(eval_logits, batch_size))
                    f2 = lambda: tf.identity(permuted_logits)
                    condition = tf.reduce_all(tf.equal(permuted_logits, tf.cast(tf.zeros([batch_size, 10]),dtype=tf.float32)))
                    permuted_logits = tf.case({condition : f1}, default=f2, exclusive=True)
                    #permuted_logits = tf.Print(permuted_logits, [permuted_logits], message='Printing permuted_logits')
                    if output_type == 'pmf':
                        loss = kl_divergence_pmf(tf.nn.softmax(permuted_logits), tf.nn.softmax(eval_logits))
                    elif output_type == 'dirichlet':
                        loss = kl_divergence_dirichlet_logits(tf.exp(permuted_logits), tf.exp(eval_logits))
                    else:
                        raise NotImplementedError("Only PMF and Dirichlet KL divergences losses are "
                                                  "currently implemented.")
                else:
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_labels, logits=eval_logits)
                grads = tf.gradients(loss, adv)[0]
                avoid_zero_div = tf.cast(1e-12, grads.dtype)
                if ord == np.inf:
                    grads = tf.sign(grads)
                elif ord == 1:
                    norm = tf.maximum(avoid_zero_div,
                                      tf.reduce_sum(tf.abs(grads), reduction_indices=[1, 2, 3], keep_dims=True))
                    grads = grads / norm

                elif ord == 2:
                    norm = tf.maximum(avoid_zero_div, tf.sqrt(
                        tf.reduce_sum(tf.square(grads), reduction_indices=[1, 2, 3], keep_dims=True)))
                    grads = grads / norm

                else:
                    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                              "currently implemented.")

                if anti_detect==True:
                    adv = adv - grads * alpha
                else:
                    adv = adv + grads * alpha
                adv = x + clip_eta(adv - x, ord, eps)
                adv = tf.clip_by_value(adv, -127.0 / 128.0, 1.0)
                adv = tf.stop_gradient(adv)
                return i + 1, adv, x, permuted_logits

            _, adversarial_images, _, _ = tf.while_loop(cond, body, [tf.zeros([]), adv, eval_images, tf.zeros([batch_size,10])], back_prop=True)

            size = self.network_architecture['n_in']
            channels = self.network_architecture['n_channels']

            count = 0
            self.sess.run(valid_iterator.initializer)
            while True:
                try:
                    labels, adv_image = self.sess.run([eval_labels, adversarial_images], feed_dict={self.dropout: 1.0})

                    writer = tf.python_io.TFRecordWriter(
                        os.path.join(save_path, 'adversarial_BIM_' + str(count) + '.tfrecord'))
                    for label, image in zip(labels, adv_image):
                        label = int(label)
                        image = np.maximum(np.minimum(np.asarray(image * 128 + 127, np.uint8), 255), 0)
                        image_raw = image.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'height': tfrecord_utils.int64_feature([size]),
                            'width': tfrecord_utils.int64_feature([size]),
                            'depth': tfrecord_utils.int64_feature([channels]),
                            'label': tfrecord_utils.int64_feature([label]),
                            'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
                        writer.write(example.SerializeToString())
                    writer.close()

                    count += 1
                except tf.errors.OutOfRangeError:
                    break

    def compute_MIM_adversarial_examples(self, eval_pattern, batch_size, eps, alpha, iter, save_path, decay_factor=1.0,
                                         ord=np.inf, anti_detect=False, output_type='pmf', least_likely=False):
        with self._graph.as_default():
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                eval_files = tf.gfile.Glob(eval_pattern)
                valid_iterator = self._construct_dataset_from_tfrecord(filenames=eval_files,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=100,
                                                                       num_threads=8,
                                                                       train=False)
                eval_labels, eval_images = valid_iterator.get_next(name='eval_data')
            momentum = tf.zeros_like(eval_images)
            adv = eval_images
            eps = eps / 128.0

            if ord == 1:
                alpha = alpha * 1024.0 * 3.0
                eps = eps * 1024.0 * 3.0
            elif ord == 2:
                alpha = alpha * 32.0 * np.sqrt(3.0)
                eps = eps * 32.0 * np.sqrt(3.0)

            def cond(i, _, __, ___, ____):
                return tf.less(i, iter)

            def body(i, adv, m, x, permuted_logits):

                with tf.variable_scope(self._model_scope, reuse=True) as scope:
                    eval_probs, eval_precision, eval_logits = self._construct_network(x=adv, keep_prob=self.dropout,
                                                                                      gain=self.network_architecture[
                                                                                          'gain'], is_training=False)

                if anti_detect:
                    if least_likely==True:
                        f1 = lambda: tf.stop_gradient(permute_logits_least_likely(eval_logits, batch_size))
                    else:
                        f1 = lambda: tf.stop_gradient(permute_logits_two_best(eval_logits, batch_size))
                    f2 = lambda: tf.identity(permuted_logits)
                    condition = tf.reduce_all(
                        tf.equal(permuted_logits, tf.cast(tf.zeros([batch_size, 10]), dtype=tf.float32)))
                    permuted_logits=tf.case({condition:f1},default=f2, exclusive=True)
                    if output_type == 'pmf':
                        loss = kl_divergence_pmf(tf.nn.softmax(permuted_logits), tf.nn.softmax(eval_logits))
                    elif output_type == 'dirichlet':
                        loss = kl_divergence_dirichlet_logits(tf.exp(permuted_logits), tf.exp(eval_logits))
                    else:
                        print 'Invalid loss!'
                        sys.exit()
                else:
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_labels, logits=eval_logits)
                grad = tf.gradients(loss, adv)[0]
                avoid_zero_div = tf.cast(1e-12, grad.dtype)

                grad = grad / tf.maximum(
                    avoid_zero_div,
                    tf.reduce_mean(tf.abs(grad), [1, 2, 3], keep_dims=True))

                m = decay_factor * m + grad

                if ord == np.inf:
                    normalized_grad = tf.sign(m)
                elif ord == 1:
                    norm = tf.maximum(avoid_zero_div, tf.reduce_sum(tf.abs(m), [1, 2, 3], keep_dims=True))
                    normalized_grad = m / norm
                elif ord == 2:
                    square = tf.reduce_sum(tf.square(m), [1, 2, 3], keep_dims=True)
                    norm = tf.sqrt(tf.maximum(avoid_zero_div, square))
                    normalized_grad = m / norm
                else:
                    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                              "currently implemented.")

                if anti_detect==True:
                    adv = adv - normalized_grad * alpha
                else:
                    adv = adv + normalized_grad * alpha

                adv = x + clip_eta(adv - x, ord, eps)
                adv = tf.clip_by_value(adv, -127.0 / 128.0, 1.0)
                adv = tf.stop_gradient(adv)
                return i + 1, adv, m, x, permuted_logits

            _, adversarial_images, _, _, _ = tf.while_loop(cond, body, [tf.zeros([]), adv, momentum, eval_images, tf.zeros([batch_size,10])],
                                                        back_prop=True)

            size = self.network_architecture['n_in']
            channels = self.network_architecture['n_channels']

            count = 0
            self.sess.run(valid_iterator.initializer)
            while True:
                try:
                    labels, adv_image = self.sess.run([eval_labels, adversarial_images], feed_dict={self.dropout: 1.0})

                    writer = tf.python_io.TFRecordWriter(
                        os.path.join(save_path, 'adversarial_MIM_' + str(count) + '.tfrecord'))
                    for label, image in zip(labels, adv_image):
                        label = int(label)
                        image = np.maximum(np.minimum(np.asarray(image * 128 + 127, np.uint8), 255), 0)
                        image_raw = image.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'height': tfrecord_utils.int64_feature([size]),
                            'width': tfrecord_utils.int64_feature([size]),
                            'depth': tfrecord_utils.int64_feature([channels]),
                            'label': tfrecord_utils.int64_feature([label]),
                            'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
                        writer.write(example.SerializeToString())
                    writer.close()

                    count += 1
                except tf.errors.OutOfRangeError:
                    break

    def compute_FGM_adversarial_examples(self, eval_pattern, batch_size, noise, save_path, anti_detect=0.0):
        with self._graph.as_default():
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                eval_files = tf.gfile.Glob(eval_pattern)
                valid_iterator = self._construct_dataset_from_tfrecord(filenames=eval_files,
                                                                       _parse_func=self.parse_func,
                                                                       _batch_func=self.batch_func,
                                                                       _map_func=self.map_func,
                                                                       batch_size=batch_size,
                                                                       capacity_mul=100,
                                                                       num_threads=8,
                                                                       train=False)
                eval_labels, eval_images = valid_iterator.get_next(name='eval_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                eval_probs, eval_precision, eval_logits = self._construct_network(x=eval_images, keep_prob=self.dropout,
                                                                                  gain=self.network_architecture[
                                                                                      'gain'], is_training=False)
            size = self.network_architecture['n_in']
            channels = self.network_architecture['n_channels']
            noise = noise / 128.0
            adversarial_images = self._FGSM_adversarial_example(eval_labels, eval_logits, eval_images, noise,
                                                                anti_detect=anti_detect)

            count = 0
            self.sess.run(valid_iterator.initializer)
            while True:
                try:
                    labels, adv_image = self.sess.run([eval_labels, adversarial_images], feed_dict={self.dropout: 1.0})

                    writer = tf.python_io.TFRecordWriter(
                        os.path.join(save_path, 'adversarial_' + str(count) + '.tfrecord'))
                    for label, image in zip(labels, adv_image):
                        label = int(label)
                        image = np.maximum(np.minimum(np.asarray(image * 128 + 127, np.uint8), 255), 0)
                        image_raw = image.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'height': tfrecord_utils.int64_feature([size]),
                            'width': tfrecord_utils.int64_feature([size]),
                            'depth': tfrecord_utils.int64_feature([channels]),
                            'label': tfrecord_utils.int64_feature([label]),
                            'image_raw': tfrecord_utils.bytes_feature([image_raw])}))
                        writer.write(example.SerializeToString())
                    writer.close()

                    count += 1
                except tf.errors.OutOfRangeError:
                    break

