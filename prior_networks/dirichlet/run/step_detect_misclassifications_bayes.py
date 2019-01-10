#! /usr/bin/env python

import argparse
import sys

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
commandLineParser.add_argument('eval_pattern', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('n_samples', type=int,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('dropout', type=float,
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


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_compute_bayesian_uncertainty.txt', 'a') as f:
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
    n_out = dpn.network_architecture['n_out']
    labels, probs, logits = dpn.eval(args.eval_pattern, batch_size=1, n_samples=args.n_samples, dropout=args.dropout)
    print labels.shape

    #Reshape into (datasize, n_samples, n_classes) shapes
    labels = np.max(np.reshape(labels, (-1, args.n_samples)), axis=1)
    probs = np.reshape(probs, (-1, args.n_samples, n_out))
    mean_probs = np.mean(probs, axis=1)
    classification_calibration(labels, mean_probs, save_path=args.output_dir)
    accuracy = jaccard_similarity_score(labels, np.argmax(mean_probs, axis=1))
    with open(os.path.join(args.output_dir, 'results.txt'), 'a') as f:
        f.write('Classification Error: ' + str(np.round(100 * (1.0 - accuracy), 1)) + '\n')

    uncertainties  = calculate_MCDP_uncertainty(probs)
    plot_accuracies(labels, mean_probs, uncertainties, save_path=args.output_dir, show=args.show)

if __name__ == '__main__':
    main()