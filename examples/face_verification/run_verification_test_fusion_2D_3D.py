import __init__
import argparse, yaml
from torch import multiprocessing as mp

import sys, os, glob, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, cal_model_parm_nums, Wandb
from openpoints.utils import EasyConfig

from run_verification_test_one_dataset import VerificationTester, parse_args



class LateFusionVerificationTester(VerificationTester):

    def __init__(self):
        super().__init__()


    def fuse_scores(self, distances1, distances2, method='mean'):
        assert distances1.shape[0] == distances2.shape[0]
        distances1 /= np.max(distances1)
        distances2 /= np.max(distances2)

        if method == 'mean':
            final_distances = (distances1 + distances2) / 2.0
        elif method == 'min':
            final_distances = np.minimum(distances1, distances2)
        elif method == 'max':
            final_distances = np.maximum(distances1, distances2)

        return final_distances


    def do_fusion_verification_test(self, model, dataset='LFW', num_points=2048, distances_pairs_2d=np.array([]), batch_size=32, fusion_methods=['mean'], verbose=True):
        model.eval()

        folds_pair_cache, folds_pair_labels, folds_indexes = self.load_organize_and_subsample_pointclouds(dataset, num_points, verbose=verbose)

        folds_pair_distances = self.compute_set_distances(model, folds_pair_cache, batch_size, verbose=verbose)
        folds_pair_distances = folds_pair_distances.cpu().detach().numpy()

        distances_fused = {}
        results_fused = {}
        for fm in fusion_methods:
            # distances_fused = self.fuse_scores(folds_pair_distances, distances_pairs_2d)
            distances_fused[fm] = self.fuse_scores(folds_pair_distances, distances_pairs_2d, method=fm)

            tpr, fpr, accuracy, tar_mean, tar_std, far_mean = self.do_k_fold_test(distances_fused[fm], folds_pair_labels, folds_indexes, verbose=verbose)
            acc_mean, acc_std = np.mean(accuracy), np.std(accuracy)

            results_fused[fm] = {}
            results_fused[fm]['acc_mean'] = acc_mean
            results_fused[fm]['acc_std'] = acc_std
            results_fused[fm]['tar_mean'] = tar_mean
            results_fused[fm]['tar_std'] = tar_std
            results_fused[fm]['far_mean'] = far_mean

        # return acc_mean, acc_std, tar_mean, tar_std, far_mean
        return results_fused



if __name__ == "__main__":

    late_fusion_verif_tester = LateFusionVerificationTester()

    # Initialization
    args, opts = parse_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if not hasattr(cfg, 'seed'):
        cfg.seed = np.random.randint(1, 10000)

    # Build model
    print('Building model...')
    model = build_model_from_cfg(cfg.model).to(0)

    # Load trained weights
    model, best_epoch, metrics = late_fusion_verif_tester.load_trained_weights_from_cfg_file(model, args.cfg)
    model.eval()

    # Load 2D distances
    if args.arcdists == 'zero' or args.arcdists == 'zeros':
        distances_pairs_2d = np.zeros((6000,))
    elif args.arcdists == 'one' or args.arcdists == 'ones':
        distances_pairs_2d = np.ones((6000,))
    elif args.arcdists == 'rand' or args.arcdists == 'random':
        distances_pairs_2d = np.random.random((6000,))
    else:
        distances_pairs_2d = np.load(args.arcdists)

    # fusion_methods = ['mean']
    # fusion_methods = ['mean', 'min']
    fusion_methods = ['mean', 'min', 'max']

    # Do fused verification test
    # acc_mean, acc_std, tar, tar_std, far = late_fusion_verif_tester.do_fusion_verification_test(model, args.dataset, args.num_points, distances_pairs_2d, args.batch, verbose=True)
    results_fused = late_fusion_verif_tester.do_fusion_verification_test(model, args.dataset, args.num_points, distances_pairs_2d, args.batch, fusion_methods, verbose=True)
    print()

    for fm in fusion_methods:
        acc_mean = results_fused[fm]['acc_mean']
        acc_std = results_fused[fm]['acc_std']
        tar_mean = results_fused[fm]['tar_mean']
        tar_std = results_fused[fm]['tar_std']
        far_mean = results_fused[fm]['far_mean']
        print('Final - dataset: %s  -  fusion_method: %s  -  acc_mean: %.6f ± %.6f  -  tar_mean: %.6f ± %.6f    far_mean: %.6f)' % (args.dataset, fm, acc_mean, acc_std, tar_mean, tar_std, far_mean))

    print('\nFinished!')
