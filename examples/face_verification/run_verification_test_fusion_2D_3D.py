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

'''
try:
    from .dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA
except ImportError as e:
    from dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA
'''

from run_verification_test_one_dataset import VerificationTester



def parse_args():
    parser = argparse.ArgumentParser('run_verification_test_fusion_2D_3D.py')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--dataset', type=str, default='lfw', help='dataset name')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points to subsample')
    parser.add_argument('--arcdists', type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/lfw_distances_arcface=1000class_acc=0.93833.npy', help='dataset name')
    
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    return args, opts


class LateFusionVerificationTester(VerificationTester):

    def __init__(self):
        super().__init__()


    def fuse_scores(self, distances1, distances2):
        distances1 /= torch.max(distances1)
        distances2 /= torch.max(distances2)
        final_distances = (distances1 + distances2) / 2.0
        return final_distances


    def do_fusion_verification_test(self, model, dataset='LFW', num_points=1200, distances_pairs_2d={}, verbose=True):
        model.eval()

        train_distances_2d = distances_pairs_2d['train']
        test_distances_2d = distances_pairs_2d['test']

        train_cache, train_pair_labels, test_cache, test_pair_labels = self.load_organize_and_subsample_pointclouds(dataset, num_points, verbose=verbose)

        train_distances_3d = self.compute_set_distances(model, train_cache, verbose=verbose)
        test_distances_3d = self.compute_set_distances(model, test_cache, verbose=verbose)

        train_distances_fused = self.fuse_scores(train_distances_3d, train_distances_2d)
        test_distances_fused = self.fuse_scores(test_distances_3d, test_distances_2d)

        best_train_tresh, best_train_acc, train_tar, train_far, \
            test_acc, test_tar, test_far, test_tp, test_fp, test_tn, test_fn = self.find_best_treshold_train_eval_test_set(train_distances_fused, train_pair_labels, test_distances_fused, test_pair_labels, verbose=verbose)

        return best_train_tresh, best_train_acc, train_tar, train_far, \
                test_acc, test_tar, test_far



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
    distances_pairs_2d = torch.from_numpy(distances_pairs_2d).float().to(0)
    # print('distances_pairs_2d:', distances_pairs_2d.device)
    # sys.exit(0)

    # Do fused verification test
    best_tresh, best_acc, tar, far = late_fusion_verif_tester.do_fusion_verification_test(model, args.dataset, args.num_points, distances_pairs_2d, verbose=True)
    print('\nFinal - dataset: %s  -  (tresh: %.6f    acc: %.6f)    (tar: %.6f    far: %.10f)' % (args.dataset, best_tresh, best_acc, tar, far))

    print('Finished!')
