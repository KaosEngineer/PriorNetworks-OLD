#! /usr/bin/env python

import argparse
import sys
import matplotlib

matplotlib.use('agg')
import numpy as np

from prior_networks.uncertainty_functions import *

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('output_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--show', type=bool, default=False,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--Log', type=bool, default=False,
                               help='which orignal data is saved should be loaded')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_evaluate_combined_blackbox_adversarial.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    measures_dict =  {'diffential_entropy' : "Differential_Entropy",
                      'entropy_expected' : "Entropy_of_Expected_Distribution",
                      'mutual_information' : "Mutual_Information"}

    flipped = np.loadtxt(os.path.join(args.output_dir,'successful_attacks.txt'), dtype=np.int32)
    success_rate = np.mean(np.asarray(flipped, dtype=np.float32))
    with open(os.path.join(args.output_dir, 'results.txt'), 'a') as f:
        f.write('Adversarial Success rate: ' + str(np.round(100 * success_rate, 1)) + '\n')

    for measure in measures_dict.keys():
        in_uncertainty = np.loadtxt(os.path.join(args.output_dir,measure + '_in.txt'), dtype=np.float32)
        out_uncertainty = np.loadtxt(os.path.join(args.output_dir,measure + '_out.txt'), dtype=np.float32)

        in_domain = np.ones_like(in_uncertainty, dtype=np.int32)
        out_domain = np.zeros_like(out_uncertainty, dtype=np.int32)
        domain_labels = np.concatenate((in_domain, out_domain), axis=0)

        plot_mod_roc_curve(domain_labels,
                           in_measure=in_uncertainty,
                           out_measure=out_uncertainty,
                           class_flipped=flipped,
                           measure_name=measures_dict[measure],
                           save_path=args.output_dir,
                           pos_label=1,
                           show=True)

        plot_histogram(uncertainty_measure=in_uncertainty,
                       measure_name=measures_dict[measure],
                       ood_uncertainty_measure=out_uncertainty,
                       save_path=args.output_dir,
                       log=False,
                       show=True,
                       bins=50)

if __name__ == '__main__':
    main()
