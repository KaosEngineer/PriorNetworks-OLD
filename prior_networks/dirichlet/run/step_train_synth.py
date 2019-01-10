#! /usr/bin/env python

import argparse
import os
import sys

import tensorflow as tf

from prior_networks.dirichlet.dirichlet_prior_network_synth import MiniPriorNet

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--batch_size', type=int, default=128,
                               help='Specify the training batch size')
commandLineParser.add_argument('--learning_rate', type=float, default=1e-3,
                               help='Specify the intial learning rate')
commandLineParser.add_argument('--lr_decay', type=float, default=0.95,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--in_domain_weight', type=float, default=0.0,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--dropout', type=float, default=1.0,
                               help='Specify the dropout keep probability')
commandLineParser.add_argument('--n_epochs', type=int, default=50,
                               help='Specify the number of epoch to run training for')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
commandLineParser.add_argument('--name', type=str, default='model',
                               help='Specify the name of the model')
commandLineParser.add_argument('--debug', type=int, choices=[0, 1, 2], default=0,
                               help='Specify the name of the model')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--gnoise', type=float, default=1.0,
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--alpha_target', type=float, default=1e3,
                               help='which orignal data is saved should be loaded')

def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_dirichlet_net.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    dpn = MiniPriorNet(network_architecture=None,
               seed=args.seed,
               name=args.name,
               save_path='./',
               load_path=args.load_path,
               debug_mode=args.debug)

    dpn.fit_gmm(gnoise=args.gnoise,
            optimizer=tf.contrib.opt.NadamOptimizer,
            optimizer_params={'beta1': 0.9, 'epsilon': 1e-8},
            alpha_target=args.alpha_target,
            in_domain_weight=args.in_domain_weight,
            learning_rate=args.learning_rate,
            lr_decay=args.lr_decay,
            dropout=args.dropout,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs)


    dpn.save()


if __name__ == '__main__':
    main()