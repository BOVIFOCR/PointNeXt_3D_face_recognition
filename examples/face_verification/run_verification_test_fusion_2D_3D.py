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

from xgboost import XGBClassifier

from run_verification_test_one_dataset import VerificationTester, LFold, parse_args



class LateFusionVerificationTester(VerificationTester):

    def __init__(self):
        super().__init__()


    def train_boosting_dec_trees_classifier(self, X_train, y_train):
        bst = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.5, objective='binary:logistic')
        bst.fit(X_train, y_train)
        return bst


    def get_boosting_scores_predictions(self, distances_pairs_2d, distances_pairs_3d, actual_issame, nrof_folds, folds_indexes, verbose=True):
        assert (distances_pairs_2d.shape[0] == distances_pairs_3d.shape[0])
        nrof_pairs = len(actual_issame)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        indices = np.arange(nrof_pairs)
        dists = np.zeros((nrof_pairs))

        distances_2d_3d = self.join_distances(distances_pairs_2d, distances_pairs_3d)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if verbose:
                print(f'training classifier - fold_idx: {fold_idx}/{nrof_folds-1}', end='\r')

            bst = self.train_boosting_dec_trees_classifier(distances_2d_3d[train_set], actual_issame[train_set])
            # predict_issame = bst.predict(distances_2d_3d[test_set])
            # print('\npredict_issame:', predict_issame)

            proba_issame = bst.predict_proba(distances_2d_3d[test_set])
            dists[test_set] = 1 - proba_issame[:,1]
            # print('\nproba_issame:', proba_issame)
            # print('\ndists[test_set]:', dists[test_set])
        
        if verbose:
            print('')

        return dists


    def join_distances(self, distances1, distances2):
        distances1 /= np.max(distances1)
        distances2 /= np.max(distances2)
        joint_distances = np.transpose(np.stack((distances1, distances2)))
        return joint_distances


    def fuse_scores(self, distances1, distances2, method='mean'):
        assert distances1.shape[0] == distances2.shape[0], f'Error, \'distances1.shape[0]\' is different of \'distances2.shape[0]\'. distances1.shape[0]={distances1.shape[0]} while distances2.shape[0]={distances2.shape[0]}'
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

        distances_pairs_3d = self.compute_set_distances(model, folds_pair_cache, batch_size, verbose=verbose)
        distances_pairs_3d = distances_pairs_3d.cpu().detach().numpy()

        distances_fused = {}
        results_fused = {}
        for fm in fusion_methods:
            if verbose:
                print(f'Fusion method: {fm}')

            if fm == 'boosting-dec-trees':
                nrof_folds=10
                distances_fused[fm] = self.get_boosting_scores_predictions(distances_pairs_2d, distances_pairs_3d, folds_pair_labels, nrof_folds, folds_indexes, verbose=verbose)

            else:  # for fusion_methods = ['mean', 'min', 'max']
                distances_fused[fm] = self.fuse_scores(distances_pairs_2d, distances_pairs_3d, method=fm)

            # tpr, fpr, accuracy, tar_mean, tar_std, far_mean = self.do_k_fold_test(distances_fused[fm], folds_pair_labels, folds_indexes, verbose=verbose)
            tpr, fpr, accuracy, tar_mean, tar_std, far_mean, \
                tp_idx, fp_idx, tn_idx, fn_idx, ta_idx, fa_idx = self.do_k_fold_test(distances_fused[fm], folds_pair_labels, folds_indexes, verbose=verbose)
            
            acc_mean, acc_std = np.mean(accuracy), np.std(accuracy)

            results_fused[fm] = {}
            results_fused[fm]['acc_mean'] = acc_mean
            results_fused[fm]['acc_std'] = acc_std
            results_fused[fm]['tar_mean'] = tar_mean
            results_fused[fm]['tar_std'] = tar_std
            results_fused[fm]['far_mean'] = far_mean
            results_fused[fm]['tp_idx'] = tp_idx
            results_fused[fm]['fp_idx'] = fp_idx
            results_fused[fm]['tn_idx'] = tn_idx
            results_fused[fm]['fn_idx'] = fn_idx
            results_fused[fm]['ta_idx'] = ta_idx
            results_fused[fm]['fa_idx'] = fa_idx

        # return acc_mean, acc_std, tar_mean, tar_std, far_mean
        return results_fused



if __name__ == "__main__":

    verif_tester = LateFusionVerificationTester()

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
    model, best_epoch, metrics = verif_tester.load_trained_weights_from_cfg_file(model, args.cfg)
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
    # fusion_methods = ['mean', 'min', 'max']
    # fusion_methods = ['boosting-dec-trees']
    fusion_methods = ['mean', 'min', 'max', 'boosting-dec-trees']

    # Do fused verification test
    # acc_mean, acc_std, tar, tar_std, far = verif_tester.do_fusion_verification_test(model, args.dataset, args.num_points, distances_pairs_2d, args.batch, verbose=True)
    results_dict = verif_tester.do_fusion_verification_test(model, args.dataset, args.num_points, distances_pairs_2d, args.batch, fusion_methods, verbose=True)
    print()

    for fm in fusion_methods:
        acc_mean = results_dict[fm]['acc_mean']
        acc_std  = results_dict[fm]['acc_std']
        tar_mean = results_dict[fm]['tar_mean']
        tar_std  = results_dict[fm]['tar_std']
        far_mean = results_dict[fm]['far_mean']
        tp_idx   = results_dict[fm]['tp_idx']
        fp_idx   = results_dict[fm]['fp_idx']
        tn_idx   = results_dict[fm]['tn_idx']
        fn_idx   = results_dict[fm]['fn_idx']
        ta_idx   = results_dict[fm]['ta_idx']
        fa_idx   = results_dict[fm]['fa_idx']

        if args.save_results:
            verif_tester.save_results(results_dict, args, model='ResNet+PointNeXt_fusion='+str(fm))

        print('Final - dataset: %s  -  fusion_method: %s  -  acc_mean: %.6f ± %.6f  -  tar_mean: %.6f ± %.6f    far_mean: %.6f)' % (args.dataset, fm, acc_mean, acc_std, tar_mean, tar_std, far_mean))

    print('\nFinished!')
