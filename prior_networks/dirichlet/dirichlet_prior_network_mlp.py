import tensorflow as tf
import tensorflow.contrib.slim as slim
from dirichlet_prior_network import PriorNet


class PriorNetMLP(PriorNet):

    def _construct_network(self, x, keep_prob, is_training, gain=False):
        if self.network_architecture['BN'] == True:
            normalizer_fn = slim.batch_norm
            normalizer_params = {'is_training': is_training,
                                 'decay': 0.97,
                                 'scale': True}
        else:
            normalizer_fn=None
            normalizer_params=None

        with tf.variable_scope('Network') as scope:
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params,
                                num_outputs=self.network_architecture['n_fhid'],
                                weights_regularizer=slim.l2_regularizer(self.network_architecture['L2'])):
                fc = slim.flatten(x)
                for layer in xrange(self.network_architecture['n_flayers']):
                    fc = slim.fully_connected(fc, scope='fc_'+str(layer))
                    fc=tf.nn.dropout(fc, keep_prob=keep_prob, seed=self._seed)

        logits = slim.fully_connected(fc, self.network_architecture['n_out'], activation_fn=None, scope='logits')
        mean = tf.nn.softmax(logits, dim=1, name='softmax_preds')
        if gain:
            gain = slim.fully_connected(fc, 1, activation_fn=tf.exp, scope='gain')
            logits = logits+gain
        alphas = tf.exp(logits)
        precision = tf.reduce_sum(alphas, axis=1, keep_dims=True)

        return mean, precision, logits