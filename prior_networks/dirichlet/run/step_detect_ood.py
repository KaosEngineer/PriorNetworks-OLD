#! /usr/bin/env python

import argparse
import sys
import matplotlib
matplotlib.use('agg')
import numpy as np

from prior_networks.uncertainty_functions import *
from prior_networks.dirichlet.dirichlet_prior_network_conv import PriorNetConv
from prior_networks.dirichlet.dirichlet_prior_network_mlp import PriorNetMLP

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
commandLineParser.add_argument('--name', type=str, default='model',
                               help='Specify the name of the model')
commandLineParser.add_argument('--debug', type=int, choices=[0, 1, 2], default=0,
                               help='Specify the name of the model')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('in_domain_pattern', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('out_domain_pattern', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('output_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('model_type',
                               choices=['MLP', 'CONV'],
                               help='which should be loaded')
commandLineParser.add_argument('--show', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--overwrite', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--Log', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--noise_corruption', type=float, default=0.0,
                               help='which orignal data is saved should be loaded')

def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_test_uncertainty.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_dir) and not args.overwrite:
        print 'Directory', args.output_dir, "exists. Exiting..."
        sys.exit()
    elif os.path.isdir(args.output_dir) and args.overwrite:
        os.remove(args.output_dir+'/*')
    else:
        os.makedirs(args.output_dir)

    if args.model_type == 'MLP':
        dpn = PriorNetMLP(network_architecture=None,
                   seed=args.seed,
                   name=args.name,
                   save_path='./',
                   fa_path=None,
                   load_path=args.load_path,
                   debug_mode=args.debug)
    elif args.model_type == 'CONV':
        dpn = PriorNetConv(network_architecture=None,
                   seed=args.seed,
                   name=args.name,
                   save_path='./',
                   fa_path=None,
                   load_path=args.load_path,
                   debug_mode=args.debug)

    in_labels, in_probs, in_logits = dpn.eval(args.in_domain_pattern, batch_size=100, n_samples=1, dropout=1.0, corruption=args.noise_corruption)
    out_labels, out_probs, out_logits = dpn.eval(args.out_domain_pattern, batch_size=100, n_samples=1, dropout=1.0, corruption=args.noise_corruption)
    in_alphas = np.exp(in_logits)
    out_alphas = np.exp(out_logits)

    print np.mean(in_logits), np.std(in_logits)
    print np.mean(out_logits), np.std(out_logits)

    #Compute Labels
    in_domain = np.zeros_like(in_labels)
    out_domain = np.ones_like(out_labels)
    domain_labels = np.concatenate((in_domain, out_domain), axis=0)

    #Calculate Uncertainties
    in_uncertainties  = calculate_dirichlet_uncertainty(in_probs, in_alphas)
    out_uncertainties = calculate_dirichlet_uncertainty(out_probs, out_alphas)

    # Plot ROC AUC Curves and Accuracy Curves
    plot_roc_curves(domain_labels, in_uncertainties, out_uncertainties, save_path=args.output_dir, log=args.Log, show=args.show)
    plot_uncertainties(in_uncertainties, out_uncertainties, save_path=args.output_dir, log=args.Log, show=args.show)

if __name__ == '__main__':
    main()
