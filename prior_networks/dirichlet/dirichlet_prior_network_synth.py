import os, sys
import time

import numpy as np
import tensorflow as tf
from scipy.stats import dirichlet
import core.utilities.utilities as util
from core.basemodel import BaseModel
import matplotlib
matplotlib.use('agg')
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.special import psi
from scipy.interpolate import griddata

import tensorflow.contrib.slim as slim

try:
    import cPickle as pickle
except:
    import pickle
from scipy.stats import norm


def mutual_information(alphas):
    alpha_0 = np.sum(alphas, axis=1)[:,np.newaxis]

    mi = -np.sum(alphas/alpha_0*(np.log(alphas/alpha_0)-psi(alphas+1.0)+psi(alpha_0+1.0)), axis=1)

    return mi

def lrelu(x):
    return tf.maximum(x, 0.2 * x)

class MiniPriorNet(BaseModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0,
                 seed=100, epoch=None):

        BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, save_path=save_path,
                           load_path=load_path, debug_mode=debug_mode)

        with self._graph.as_default():
            with tf.variable_scope('input') as scope:
                size = self.network_architecture['n_in']
                self._input_scope = scope
                self.batch_size = tf.placeholder(tf.int32, [])
                self.dropout = tf.placeholder(tf.float32, [])
                self.gnoise = tf.placeholder(tf.float32, [])
                self.images = tf.placeholder(tf.float32, [None, size])  # Can I specify this automatically??

            with tf.variable_scope('PriorNet') as scope:
                # Not sure if this is even really necessary....
                self.gmm_X, self.gmm_Y, self.gmm_noise = self._sample_gmm(batch_size=self.batch_size, noise=self.gnoise)

                self._model_scope = scope
                self.mean, self.precision, self.logits = self._construct_network(x=self.images, keep_prob=self.dropout,
                                                                                 is_training=False)

            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if load_path is None:
            with self._graph.as_default():
                init = tf.global_variables_initializer()
                self.sess.run(init)

                # If necessary, restore model from previous
        elif load_path is not None:
            self.load(load_path=load_path, step=epoch)


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

    def _construct_network(self, x, keep_prob, is_training):

        with tf.variable_scope('Network') as scope:
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                num_outputs=self.network_architecture['n_fhid'],
                                weights_regularizer=slim.l2_regularizer(self.network_architecture['L2'])):
                fc = slim.flatten(x)
                for layer in xrange(self.network_architecture['n_flayers']):
                    fc = slim.fully_connected(fc, scope='fc_'+str(layer))
                    fc=tf.nn.dropout(fc, keep_prob=keep_prob, seed=self._seed)

        logits = slim.fully_connected(fc, self.network_architecture['n_out'], activation_fn=None, scope='logits')

        alphas = tf.exp(logits)
        precision = tf.reduce_sum(alphas, axis=1, keep_dims=True)
        mean = tf.nn.softmax(logits, dim=1, name='softmax_preds')

        return mean, precision, logits

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

            eval_data = tf.random_uniform(shape=[30000, 2], minval=-20, maxval=20.0, dtype=tf.float32, seed=self._seed)
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                mean, precision, logits = self._construct_network(x=self.gmm_X,
                                                                  keep_prob=self.dropout,
                                                                  is_training=True)
                noise_mean, noise_precision, noise_logits = self._construct_network(x=self.gmm_noise,
                                                                                    keep_prob=self.dropout,
                                                                                    is_training=True)
                eval_mean, eval_prevision, eval_logits = self._construct_network(x=eval_data,
                                                                                 keep_prob=1.0,
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
            mi = mutual_information(alphas)

            xi = np.linspace(-20, 20, 10000)
            yi = np.linspace(-20, 20, 10000)
            #points = np.concatenate([xi,yi], axis=1)
            #zi = griddata((x, y), z, , method='linear')
            zi_entropy = griddata(inputs, entropy, (xi[None, :], yi[:, None]), method='cubic')
            zi_diff_entropy = griddata(inputs, diff_entropy , (xi[None, :], yi[:, None]), method='cubic')
            zi_mutual_information = griddata(inputs, mi, (xi[None, :], yi[:, None]), method='cubic')
            print zi_entropy.shape
            plt.contourf(xi, yi,zi_entropy, cmap=cm.Blues, alpha=0.9)
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.colorbar()
            plt.savefig('Entropy.png', bbox_inches='tight')
            plt.close()
            plt.contourf(xi, yi,zi_mutual_information, cmap=cm.Blues, alpha=0.9)
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.colorbar()
            plt.savefig('mutual_information.png', bbox_inches='tight')
            plt.close()
            plt.contourf(xi, yi, zi_diff_entropy, cmap=cm.Blues, alpha=0.9)
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.colorbar()
            plt.savefig('DEntropy.png', bbox_inches='tight')
            plt.close()

            duration = start_time - time.time()
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % duration + '\n')
                f.write('----------------------------------------------------------\n')
            print (format_str % duration)
