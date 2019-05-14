#! /usr/bin/env python

import argparse
import os
import sys



commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--batch_size', type=int, default=128,
                               help='Specify the training batch size')
commandLineParser.add_argument('--learning_rate', type=float, default=1e-3,
                               help='Specify the intial learning rate')
commandLineParser.add_argument('--momentum', type=float, default=0.9,
                               help='Specify the intial learning rate')
commandLineParser.add_argument('--beta2', type=float, default=0.999,
                               help='Specify the intial learning rate')
commandLineParser.add_argument('--lr_decay', type=float, default=0.95,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--ce_weight', type=float, default=0.0,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--noise_weight', type=float, default=1.0,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--dropout', type=float, default=1.0,
                               help='Specify the dropout keep probability')
commandLineParser.add_argument('--n_epochs', type=int, default=50,
                               help='Specify the number of epoch to run training for')
commandLineParser.add_argument('--cycle_length', type=int, default=30,
                               help='Specify the number of epoch to run training for')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
commandLineParser.add_argument('--name', type=str, default='model',
                               help='Specify the name of the model')
commandLineParser.add_argument('--debug', type=int, choices=[0, 1, 2], default=0,
                               help='Specify the name of the model')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--noise', type=bool, default=False,
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('data_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('data_size', type=int,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('valid_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('loss',
                               choices=['KL', 'NLL-MI', 'NLL-DE'],
                               help='which should be loaded')
commandLineParser.add_argument('mode',
                               choices=['OOD', 'ID'],
                               help='which should be loaded')
commandLineParser.add_argument('--gpu', type=int, default=0,
                               help='Specify path to which model should be saved')
commandLineParser.add_argument('--smoothing', type=float, default=1e-2,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--fa_path', type=str, default=None,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--alpha_target', type=float, default=1e3,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--augment', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--noise_path', type=str, default=None,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('model_type',
                               choices=['MLP', 'CONV'],
                               help='which should be loaded')
def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_dirichlet_net.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    print args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import tensorflow as tf
    from prior_networks.dirichlet.dirichlet_prior_network_conv import PriorNetConv
    from prior_networks.dirichlet.dirichlet_prior_network_mlp import PriorNetMLP

    if args.model_type == 'MLP':
        dpn = PriorNetMLP(network_architecture=None,
                   seed=args.seed,
                   name=args.name,
                   save_path='./',
                   fa_path=args.fa_path,
                   load_path=args.load_path,
                   debug_mode=args.debug)
    elif args.model_type == 'CONV':
        dpn = PriorNetConv(network_architecture=None,
                   seed=args.seed,
                   name=args.name,
                   save_path='./',
                   fa_path=args.fa_path,
                   load_path=args.load_path,
                   debug_mode=args.debug)

    if args.noise is True:
        dpn.fit_with_noise(train_pattern=args.data_path,
                           valid_pattern=args.valid_path,
                           n_examples=args.data_size,
                           alpha_target=args.alpha_target,
                           noise_pattern=args.noise_path,
                           cycle_length=args.cycle_length,
                           mode=args.mode,
                           smoothing=args.smoothing,
                           augment=args.augment, loss=args.loss,
                           learning_rate=args.learning_rate,
                           lr_decay=args.lr_decay, dropout=args.dropout,
                           ce_weight=args.ce_weight,
                           noise_weight=args.noise_weight,
                           batch_size=args.batch_size,
                           optimizer=tf.contrib.opt.NadamOptimizer,
                           optimizer_params={'beta1': args.momentum, 'beta2': args.beta2, 'epsilon': 1e-8},
                           n_epochs=args.n_epochs)
    else:
        dpn.fit(train_pattern=args.data_path,
                valid_pattern=args.valid_path,
                n_examples=args.data_size,
                cycle_length=args.cycle_length,
                #optimizer=tf.train.MomentumOptimizer,
                #optimizer_params={'momentum': 0.9, 'use_nesterov': True},
                optimizer=tf.contrib.opt.NadamOptimizer,
                optimizer_params={'beta1': args.momentum, 'beta2': args.beta2, 'epsilon': 1e-8},
                learning_rate=args.learning_rate,
                augment=args.augment,
                lr_decay=args.lr_decay,
                dropout=args.dropout,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs)

    dpn.save()


if __name__ == '__main__':
    main()