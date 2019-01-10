#! /usr/bin/env python
import argparse
import os
import sys
try:
    import cPickle as pickle
except:
    import pickle


from prior_networks.dirichlet.dirichlet_prior_network_synth import MiniPriorNet

from core.utilities.utilities import activation_dict
from core.utilities.utilities import initializer_dict
from core.utilities.utilities import activation_fn_list
from core.utilities.utilities import initializer_list

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path location wheree to setup ')
commandLineParser.add_argument('library_path', type=str,
                               help='absolute path location wheree to setup ')
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
commandLineParser.add_argument('--n_in', type=int, default=28,
                               help='which should be loaded')
commandLineParser.add_argument('--n_fhid', type=int, default=50,
                               help='which should be loaded')
commandLineParser.add_argument('--n_flayers', type=int, default=1,
                               help='which should be loaded')
commandLineParser.add_argument('--n_out', type=int, default=10,
                               help='which should be loaded')
commandLineParser.add_argument('--l2', type=float, default=0.0,
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
    os.symlink(args.library_path, 'MalLib')

    #Define network architecture
    network_architecture=dict(model_name=args.name,         # Define Model Type
                              n_in=args.n_in,               # data input dimension
                              n_fhid=args.n_fhid,           # feed-forward hidden layer size
                              n_flayers=args.n_flayers,     # Number of feed-forward hidden layers
                              n_out=args.n_out,             # output dimensionality
                              f_activation_fn=activation_dict[args.f_activation_fn], # Activation Function feed-forward layers
                              output_fn=activation_dict[args.output_fn],             # Output function
                              initializer=initializer_dict[args.initializer],        # Parameter Initializer
                              L2=args.l2,                   # L2 weight decay
                              init_params = {'factor':args.factor, 'uniform':args.uniform, 'mode':args.mode}  # Base number of conv filters
    )

    # initialize the model and save intialized parameters

    dpn = MiniPriorNet(network_architecture=network_architecture,
                   seed=args.seed,
                   name=args.name,
                   save_path=args.save_path,
                   load_path=args.load_path,
                   debug_mode=args.debug)

    dpn.save()

if __name__ == '__main__':
    main()