#! /usr/bin/env python

import argparse
import sys
import os

from prior_networks.dirichlet.dirichlet_prior_network_conv import PriorNetConv
from prior_networks.dirichlet.dirichlet_prior_network_mlp import PriorNetMLP
import numpy as np

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
commandLineParser.add_argument('output_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('attack',
                               choices=['FGM', 'BIM', 'MIM'],
                               help='which should be loaded')
commandLineParser.add_argument('noise', type=float,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('model_type',
                               choices=['MLP', 'CONV'],
                               help='which should be loaded')
commandLineParser.add_argument('--anti_detect', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--output_type', type=str, default='pmf',
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--least_likely', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--ord', type=float, default=np.inf,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--iter', type=int, default=10,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--show', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--overwrite', type=bool, default=False,
                               help='which orignal data is saved should be loaded')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_compute_adversarial_examples.txt', 'a') as f:
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

    if args.attack == 'FGM':
        dpn.compute_FGM_adversarial_examples(args.eval_pattern,
                                             batch_size=100,
                                             noise=args.noise,
                                             save_path = args.output_dir,
                                             anti_detect=args.anti_detect)
    elif args.attack == 'BIM':
        dpn.compute_BIM_adversarial_examples(args.eval_pattern,
                                             batch_size=100,
                                             eps=args.noise,
                                             alpha=args.noise/(128*args.iter),
                                             iter=args.iter,
                                             ord=args.ord,
                                             save_path=args.output_dir,
                                             anti_detect=args.anti_detect,
                                             least_likely=args.least_likely,
                                             output_type=args.output_type)
    elif args.attack == 'MIM':
        dpn.compute_MIM_adversarial_examples(args.eval_pattern,
                                             batch_size=100,
                                             eps=args.noise,
                                             alpha=args.noise/(128*args.iter),
                                             iter=args.iter,
                                             ord=args.ord,
                                             save_path=args.output_dir,
                                             anti_detect=args.anti_detect,
                                             least_likely=args.least_likely,
                                             output_type=args.output_type)
    else:
        raise NotImplementedError("Only FGM, BIM and MIM attacks are "
                                  "currently implemented.")



if __name__ == '__main__':
    main()
