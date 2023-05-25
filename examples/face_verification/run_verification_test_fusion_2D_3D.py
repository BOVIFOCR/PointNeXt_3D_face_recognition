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


    def fuse_scores(self, distances1, distances2):
        assert distances1.shape[0] == distances2.shape[0]
        distances1 /= np.max(distances1)
        distances2 /= np.max(distances2)
        final_distances = (distances1 + distances2) / 2.0
        return final_distances


    def do_fusion_verification_test(self, model, dataset='LFW', num_points=2048, distances_pairs_2d=np.array([]), batch_size=32, verbose=True):
        model.eval()

        # train_distances_2d = distances_pairs_2d['train']
        # test_distances_2d = distances_pairs_2d['test']

        folds_pair_cache, folds_pair_labels, folds_indexes = self.load_organize_and_subsample_pointclouds(dataset, num_points, verbose=verbose)

        folds_pair_distances = self.compute_set_distances(model, folds_pair_cache, batch_size, verbose=verbose)
        folds_pair_distances = folds_pair_distances.cpu().detach().numpy()

        distances_fused = self.fuse_scores(folds_pair_distances, distances_pairs_2d)

        _, _, accuracy, val, val_std, far = self.do_k_fold_test(distances_fused, folds_pair_labels, folds_indexes, verbose=verbose)
        acc_mean, acc_std = np.mean(accuracy), np.std(accuracy)

        return acc_mean, acc_std



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
    distances_pairs_2d = np.load(args.arcdists)
    # distances_pairs_2d = torch.from_numpy(distances_pairs_2d).float().to(0)
    # print('distances_pairs_2d:', distances_pairs_2d.device)
    # sys.exit(0)

    # Do fused verification test
    acc_mean, acc_std = late_fusion_verif_tester.do_fusion_verification_test(model, args.dataset, args.num_points, distances_pairs_2d, args.batch, verbose=True)

    print('\nFinal - dataset: %s  -  acc_mean: %.6f    acc_std: %.6f)' % (args.dataset, acc_mean, acc_std))
    print('Finished!')
