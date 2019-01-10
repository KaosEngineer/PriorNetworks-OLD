#! /usr/bin/env python
import argparse
import os
import sys
try:
    import cPickle as pickle
except:
    import pickle

from prior_networks.dirichlet.dirichlet_prior_network_conv import PriorNetConv
from prior_networks.dirichlet.dirichlet_prior_network_mlp import PriorNetMLP
from core.utilities.utilities import activation_dict
from core.utilities.utilities import initializer_dict
from core.utilities.utilities import activation_fn_list
from core.utilities.utilities import initializer_list

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('data_path', type=str,
                               help='absolute path to data')
commandLineParser.add_argument('--fa_data_path', type=str, default=None,
                               help='absolute path to FA data')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path location wheree to setup ')
commandLineParser.add_argument('library_path', type=str,
                               help='absolute path location wheree to setup ')
commandLineParser.add_argument('model_type',
                               choices=['MLP', 'CONV'],
                               help='which should be loaded')
commandLineParser.add_argument('--name', type=str, default='DPN',
                               help='Specify the name of the model')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
commandLineParser.add_argument('--debug', type=int, choices=[0, 1, 2], default=0,
                               help='Specify the debug output level')
commandLineParser.add_argument('--load_path', type=str, default=None,
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--save_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--init', type=str, default=None,
                               help='Specify path to from which to initialize model')
commandLineParser.add_argument('--epoch', type=str, default=None,
                               help='which should be loaded')
commandLineParser.add_argument('--n_z', type=int, default=50,
                               help='which should be loaded')
commandLineParser.add_argument('--n_in', type=int, default=28,
                               help='which should be loaded')
commandLineParser.add_argument('--n_channels', type=int, default=1,
                               help='which should be loaded')
commandLineParser.add_argument('--n_fhid', type=int, default=50,
                               help='which should be loaded')
commandLineParser.add_argument('--n_fsize', type=int, default=3,
                               help='which should be loaded')
commandLineParser.add_argument('--n_flayers', type=int, default=1,
                               help='which should be loaded')
commandLineParser.add_argument('--n_filters', type=int, default=16,
                               help='which should be loaded')
commandLineParser.add_argument('--n_depth', type=int, default=1,
                               help='which should be loaded')
commandLineParser.add_argument('--n_clayers', type=int, default=1,
                               help='which should be loaded')
commandLineParser.add_argument('--n_out', type=int, default=10,
                               help='which should be loaded')
commandLineParser.add_argument('--l2', type=float, default=0.0,
                               help='which should be loaded')
commandLineParser.add_argument('--BN', type=bool, default=False,
                               help='which should be loaded')
commandLineParser.add_argument('--LN', type=bool, default=False,
                               help='which should be loaded')
commandLineParser.add_argument('--gain', type=bool, default=False,
                               help='which should be loaded')
commandLineParser.add_argument('--factor', type=int, default=1.0,
                               help='which should be loaded')
commandLineParser.add_argument('--uniform', type=bool, default=False,
                               help='which should be loaded')
commandLineParser.add_argument('--mode', type=str, default='FAN_IN',
                               help='which should be loaded')
commandLineParser.add_argument('--f_activation_fn',
                               choices=activation_fn_list,
                               default='lrelu',
                               help='which should be loaded')
commandLineParser.add_argument('--output_fn', choices=activation_fn_list,
                                default='sigmoid',
                                help='which should be loaded')
commandLineParser.add_argument('--initializer', choices=initializer_list,
                                default='xavier',
                                help='which should be loaded')


def main(argv=None):
    """Converts a dataset to tfrecords."""
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_setup_prior_network.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if os.path.isdir(args.destination_dir):
        print 'destination directory exists. Exiting...'
    else:
        os.makedirs(args.destination_dir)

    # Link and and create directories
    os.chdir(args.destination_dir)
    os.mkdir('model')
    os.symlink(args.data_path, 'data')
    os.symlink(args.library_path, 'MalLib')

    #Define network architecture
    network_architecture=dict(model_name=args.name,         # Define Model Type
                              n_z=args.n_z,                 # latent representation dimensionality dimension
                              n_in=args.n_in,               # data input dimension
                              n_depth=args.n_depth,         # Depth of DenseNet Layers
                              n_channels=args.n_channels,   # Number of color channels
                              n_fhid=args.n_fhid,           # feed-forward hidden layer size
                              n_flayers=args.n_flayers,     # Number of feed-forward hidden layers
                              n_out=args.n_out,             # output dimensionality
                              f_activation_fn=activation_dict[args.f_activation_fn], # Activation Function feed-forward layers
                              output_fn=activation_dict[args.output_fn],             # Output function
                              initializer=initializer_dict[args.initializer],        # Parameter Initializer
                              L2=args.l2,                   # L2 weight decay
                              BN=args.BN,                   # Use batchnorm
                              LN=args.LN,                   # Use  layernorms
                              gain = args.gain,             # Use
                              n_fsize=args.n_fsize,           # Conv filter size
                              n_clayers=args.n_clayers,     # Number of Conv Layers
                              n_filters=args.n_filters,      # Base number of conv filters
                              init_params = {'factor':args.factor, 'uniform':args.uniform, 'mode':args.mode}  # Base number of conv filters
    )

    # initialize the model and save intialized parameters
    if args.model_type == 'MLP':
        dpn = PriorNetMLP(network_architecture=network_architecture,
                       seed=args.seed,
                       name=args.name,
                       save_path=args.save_path,
                       load_path=args.load_path,
                       fa_path=args.fa_data_path,
                       debug_mode=args.debug)
    elif args.model_type == 'CONV':
        dpn = PriorNetConv(network_architecture=network_architecture,
                       seed=args.seed,
                       name=args.name,
                       save_path=args.save_path,
                       load_path=args.load_path,
                       fa_path=args.fa_data_path,
                       debug_mode=args.debug)
    else:
        print 'Incorrect Model Type. Exiting...'
        sys.exit()
    dpn.save()

if __name__ == '__main__':
    main()