import tensorflow as tf
import tensorflow.contrib.slim as slim
from dirichlet_prior_network import PriorNet

def lrelu(x):
    return tf.maximum(x, 0.2 * x)


class PriorNetConv(PriorNet):

    def _construct_network(self, x, keep_prob, is_training, gain=False):
        if self.network_architecture['BN'] == True:
            normalizer_fn = slim.batch_norm
            normalizer_params = {'is_training': is_training,
                                 'decay': 0.97,
                                 'scale': True,
                                 'fused': True}
        else:
            normalizer_fn=None
            normalizer_params=None

        with tf.variable_scope('Network') as scope:
            n_filters=self.network_architecture['n_filters']
            conv = x
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=self.network_architecture['f_activation_fn'],
                                normalizer_fn=normalizer_fn,
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                   mode='FAN_IN',
                                                                                                   uniform=False),
                                normalizer_params=normalizer_params,
                                weights_regularizer=slim.l2_regularizer(self.network_architecture['L2'])):
                conv_keep_prob = tf.minimum(keep_prob+0.3, 1.0)
                with slim.arg_scope([slim.conv2d],
                                    kernel_size=[3, 3],
                                    stride=1, padding='SAME'):
                    for layer in xrange(self.network_architecture['n_clayers']):
                        filters=min([n_filters*(layer+1), 512])
                        conv = slim.conv2d(conv, filters, scope='conv_'+str(layer))
                        conv=tf.nn.dropout(conv, keep_prob=conv_keep_prob, seed=self._seed)
                        conv = slim.conv2d(conv, filters, scope='conv2_'+str(layer))
                        if layer > 2:
                            conv = tf.nn.dropout(conv, keep_prob=conv_keep_prob, seed=self._seed)
                            conv = slim.conv2d(conv, filters, scope='conv3_' + str(layer))
                            conv = tf.nn.dropout(conv, keep_prob=conv_keep_prob, seed=self._seed)
                            conv = slim.conv2d(conv, filters, scope='conv4_' + str(layer))
                        conv = slim.max_pool2d(conv, [2,2])


                with slim.arg_scope([slim.fully_connected],
                                    num_outputs=self.network_architecture['n_fhid'],
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(**self.network_architecture['init_params'])):
                                    #weights_initializer=tf.initializers.random_normal(stddev=0.01)):
                    fc = slim.flatten(conv)
                    for layer in xrange(self.network_architecture['n_flayers']):
                        fc = slim.fully_connected(fc, scope='fc_'+str(layer))
                        fc=tf.nn.dropout(fc, keep_prob=keep_prob, seed=self._seed)

            logits = slim.fully_connected(fc, self.network_architecture['n_out'], activation_fn=None, scope='logits')
            mean = tf.nn.softmax(logits, dim=1, name='softmax_preds')
            if gain:
                gain = slim.fully_connected(fc, 1, activation_fn=tf.nn.relu, scope='gain')
                logits = logits+gain

            alphas = tf.exp(logits)
            precision = tf.reduce_sum(alphas, axis=1, keep_dims=True)


        return mean, precision, logits