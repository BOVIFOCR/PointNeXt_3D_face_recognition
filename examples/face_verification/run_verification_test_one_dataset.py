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
from sklearn.model_selection import KFold
import sklearn
from scipy import interpolate

try:
    from .dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA
except ImportError as e:
    from dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA



def parse_args():
    parser = argparse.ArgumentParser('run_verification_test.py')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--dataset', type=str, default='lfw', help='dataset name')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points to subsample')
    # parser.add_argument('--pretrained', type=str, required=True, help='checkpoint_file.pth')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    return args, opts


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


class VerificationTester:
    
    def __init__(self):
        self.LFW_POINT_CLOUDS = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw'
        self.LFW_BENCHMARK_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairs.txt'        # benchmark test set (6000 face pairs)

        self.MLFW_POINT_CLOUDS = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MLFW'
        # MLFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/MLFW/pairs.txt'


    def pc_normalize(self, pc):
        pc /= 100
        pc = (pc - pc.min()) / (pc.max() - pc.min())
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


    def load_trained_weights_from_cfg_file(self, model, cfg_path='log/ms1mv2_3d_arcface/ms1mv2_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6520-20230429-170750-98s4kHjBdbgbRWcegn9V9y/pointnext-s_arcface.yaml'):
        # pretrained_path = 'log/ms1mv2_3d_arcface/ms1mv2_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6520-20230429-170750-98s4kHjBdbgbRWcegn9V9y/checkpoint/ms1mv2_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6520-20230429-170750-98s4kHjBdbgbRWcegn9V9y_ckpt_best.pth'
        file_name = '*_ckpt_best.pth'
        # file_name = '*_ckpt_latest.pth'
        pretrained_path = '/'.join(cfg_path.split('/')[:-1]) + '/checkpoint/' + file_name
        full_pretrained_path = glob.glob(pretrained_path)
        if len(full_pretrained_path) > 0:
            full_pretrained_path = full_pretrained_path[0]
        print('Loading trained weights:', full_pretrained_path)
        best_epoch, metrics = load_checkpoint(model, full_pretrained_path)
        return model, best_epoch, metrics


    def load_one_point_cloud(self, file_path):
        if file_path.endswith('.npy'):
            points = np.load(file_path).astype(np.float32)
        points[:, :3] = self.pc_normalize(points[:, :3])
        return points


    def load_point_clouds_from_disk(self, pairs_paths, verbose=True):
        dataset = [None] * len(pairs_paths)
        for i, pair in enumerate(pairs_paths):
            pair_label, path0, path1 = pair

            if verbose:
                # print(f'pair: {i}/{len(pairs_paths)-1}', end='\r')
                print('loading_point_clouds_from_disk')
                print(f'pair: {i}/{len(pairs_paths)-1}')
                print('pair_label:', pair_label)
                print('path0:', path0)
                print('path1:', path1)

            pc0 = np.load(path0)
            pc1 = np.load(path1)
            pc0[:, :3] = self.pc_normalize(pc0[:, :3])
            pc1[:, :3] = self.pc_normalize(pc1[:, :3])

            # dataset[i] = (label, pc0, pc1)
            dataset[i] = (pc0, pc1, pair_label)

            if verbose:
                print('------------')
        if verbose:
            print()
        return dataset


    def load_dataset(self, dataset_name='lfw', verbose=True):
        if dataset_name.upper() == 'LFW':
            file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
            all_pairs_paths_label, folds_indexes, pos_pair_label, neg_pair_label = LFW_Pairs_3DReconstructedMICA().load_pointclouds_pairs_with_labels(self.LFW_POINT_CLOUDS, self.LFW_BENCHMARK_VERIF_PAIRS_LIST, file_ext)

            if verbose:
                # print('\nLFW_Pairs_3DReconstructedMICA - load_pointclouds_pairs_with_labels')
                # print('all_pairs_paths_label:', all_pairs_paths_label)
                # print('len(all_pairs_paths_label):', len(all_pairs_paths_label))
                # print('pos_pair_label:', pos_pair_label)
                # print('neg_pair_label:', neg_pair_label)
                print('Loading dataset:', dataset_name)

            folds_pair_data = self.load_point_clouds_from_disk(all_pairs_paths_label, verbose=verbose)
            folds_pair_labels = np.array([int(folds_pair_data[i][2]) for i in range(len(folds_pair_data))])   # folds_data[i] is (pc0, pc1, pair_label)
            
        else:
            print(f'\nError: dataloader for dataset \'{dataset_name}\' not implemented!\n')
            sys.exit(0)

        return folds_pair_data, folds_pair_labels, folds_indexes


    def subsample_point_cloud(self, points, npoints=1024):
        points = torch.from_numpy(points).float().to(0)
        points = torch.unsqueeze(points, dim = 0)
        # print('points:', points)
        # print('points.shape:', points.shape)

        # Copied from 'train_one_epoch()' method
        num_curr_pts = points.shape[1]
        # npoints = args.num_points
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                # raise NotImplementedError()  # original
                point_all = npoints            # Bernardo
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(points[:, :, :3].contiguous(), point_all)
            # fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = torch.gather(points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        return points

        # data = {}
        # data['pos'] = points[:, :, :3].contiguous()
        # data['x'] = points[:, :, :3].transpose(1, 2).contiguous()
        # return data


    def organize_and_subsample_pointcloud(self, dataset, npoints=1200, verbose=True):
        chanels = 3
        cache = {}
        # cache['pos'] = torch.zeros(len(dataset), chanels, npoints)
        cache['x'] = torch.zeros(size=(2*len(dataset), npoints, chanels), device=0)
        for i, pair in enumerate(dataset):
            pc0, pc1, pair_label = pair
            pc0_orig_shape, pc1_orig_shape = pc0.shape, pc1.shape
            pc0 = self.subsample_point_cloud(pc0[:, :3], npoints)
            pc1 = self.subsample_point_cloud(pc1[:, :3], npoints)
            j = i * 2
            cache['x'][j]   = pc0
            cache['x'][j+1] = pc1

            if verbose:
                print(f'organize_and_subsample_pointcloud - pair {i}/{len(dataset)-1} - pc0: {pc0_orig_shape} ->', pc0.size(), f',  pc1: {pc1_orig_shape} ->', pc1.size())
        # print('cache[\'x\'].size():', cache['x'].size())
        
        cache['pos'] = cache['x'].contiguous()
        cache['x'] = cache['x'].transpose(1, 2).contiguous()
        return cache


    def compute_embeddings_distance_insightface(self, face_embedd, verbose=True):
        assert face_embedd.size()[0] % 2 == 0
        distances = torch.zeros(int(face_embedd.size()[0]/2))
        for i in range(0, face_embedd.size()[0], 2):
            embedd0, embedd1 = face_embedd[i], face_embedd[i+1]
            distances[int(i/2)] = torch.sum( torch.square( F.normalize(torch.unsqueeze(embedd0, 0)) - F.normalize(torch.unsqueeze(embedd1, 0)) ) )
        # if verbose:
        #     print('distances:', distances)
        #     print('distances.size():', distances.size())
        return distances
    

    def compute_embeddings_cosine_distance(self, face_embedd, verbose=True):
        assert face_embedd.size()[0] % 2 == 0
        distances = torch.zeros(int(face_embedd.size()[0]/2))
        for i in range(0, face_embedd.size()[0], 2):
            embedd0, embedd1 = face_embedd[i], face_embedd[i+1]
            # distances[int(i/2)] = torch.sum( torch.square( F.normalize(torch.unsqueeze(embedd0, 0)) - F.normalize(torch.unsqueeze(embedd1, 0)) ) )
            distances[int(i/2)] = 1 - torch.dot(embedd0, embedd1)/(torch.linalg.norm(embedd0)*torch.linalg.norm(embedd1))
        # if verbose:
        #     print('distances:', distances)
        #     print('distances.size():', distances.size())
        return distances


    def eval_one_treshold(self, cos_dists, pair_labels, tresh=2.0, verbose=True):
        tp, fp, tn, fn = 0., 0., 0., 0.
        for j, (cos_dist, pair_label) in enumerate(zip(cos_dists, pair_labels)):
            if pair_label == 1:   # positive pair
                if cos_dist < tresh:
                    tp += 1
                else:
                    fn += 1
            else:  # negative pair
                if cos_dist >= tresh:
                    tn += 1
                else:
                    fp += 1

        acc = round((tp + tn) / (tp + tn + fp + fn), 4)
        tar = float(tp) / (float(tp) + float(fn))
        far = float(fp) / (float(fp) + float(tn))

        if verbose:
            print('tresh: %.6f    acc: %.6f    tar: %.6f    far: %.6f' % (tresh, acc, tar, far))

        return tp, fp, tn, fn, acc, tar, far


    def get_coeficient_and_exponent(self, desired_far=1e-03):
        coeficient = desired_far
        expoent = 0
        while coeficient < 1:
            expoent += 1
            coeficient *= 10
        return coeficient, expoent


    def find_best_treshold(self, cos_dists, pair_labels, verbose=True):
        best_tresh = 0
        best_acc = 0
        
        # start, end, step = 0, 4, 0.001   # used in insightface code: https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval/verification.py#L181
        start, end, step = 0, 2, 0.001
        # start, end, step = 0, 4, 0.005
        # start, end, step = 0, 1, 0.005

        all_margins_eval = torch.arange(start, end+step, step, dtype=torch.float64)
        all_tp_eval = torch.zeros_like(all_margins_eval, dtype=torch.float64)
        all_fp_eval = torch.zeros_like(all_margins_eval, dtype=torch.float64)
        all_tn_eval = torch.zeros_like(all_margins_eval, dtype=torch.float64)
        all_fn_eval = torch.zeros_like(all_margins_eval, dtype=torch.float64)
        all_acc_eval = torch.zeros_like(all_margins_eval, dtype=torch.float64)
        all_tar_eval = torch.zeros_like(all_margins_eval, dtype=torch.float64)
        all_far_eval = torch.zeros_like(all_margins_eval, dtype=torch.float64)

        pair_labels = torch.tensor(pair_labels, dtype=torch.int8)

        treshs = torch.arange(start, end+step, step)
        for i, tresh in enumerate(treshs):

            all_tp_eval[i], all_fp_eval[i], all_tn_eval[i], all_fn_eval[i], all_acc_eval[i], all_tar_eval[i], all_far_eval[i] = self.eval_one_treshold(cos_dists, pair_labels, tresh, verbose)

            if verbose:
                print('\x1b[2K', end='')
                print(f'tester_multitask_FACEVERIFICATION - {i}/{len(treshs)-1} - tresh: {tresh}', end='\r')

        best_acc_idx = torch.argmax(all_acc_eval)
        best_acc = all_acc_eval[best_acc_idx]
        best_tresh = treshs[best_acc_idx]

        desired_far = 1e-03
        coeficient, expoent = self.get_coeficient_and_exponent(desired_far)
        coeficient += 1
        next_far = coeficient / (10 ** expoent)   # for far=0.001, next_far=0.002
         
        desired_far_idx = torch.where(all_far_eval < next_far)[0][-1]
        tar = all_tar_eval[desired_far_idx]
        far = all_far_eval[desired_far_idx] if all_far_eval[desired_far_idx] > .0 else desired_far

        return best_tresh, best_acc, tar, far
        # return best_tresh, best_acc, tar, desired_far


    def compute_set_distances(self, model, cache={}, verbose=True):
        if verbose:
            print('cache[\'x\'].size():', cache['x'].size())

        with torch.no_grad():
            distances = torch.zeros(int(len(cache['x'])/2))
            batch_size = 64
            num_batches = len(cache['x']) // batch_size
            last_batch_size = len(cache['x']) % batch_size
            if last_batch_size > 0: num_batches += 1
            for i in range(0, num_batches):
                start_batch_idx = i*batch_size
                num_samples = batch_size
                if start_batch_idx+batch_size > len(cache['x']): num_samples = last_batch_size
                end_batch_idx = start_batch_idx+num_samples

                data = {}
                data['pos'] = cache['pos'][start_batch_idx:end_batch_idx]
                data['x']   = cache['x'][start_batch_idx:end_batch_idx]
                embedd = model.get_face_embedding(data)

                if verbose:
                    # print('embedd:', embedd)
                    print(f'computing face embeddings - batch_size: {batch_size} - batch {i}/{num_batches-1} - batch_idxs: {start_batch_idx}:{end_batch_idx} - embedd.size():', embedd.size())

                if verbose:
                    print('computing distances')
                
                dist = self.compute_embeddings_distance_insightface(embedd, verbose=verbose)
                # dist = self.compute_embeddings_cosine_distance(embedd, verbose=verbose)
                
                if verbose:
                    print('dist.size():', dist.size())
                
                distances[int(start_batch_idx/2):int(end_batch_idx/2)] = dist

                if verbose:
                    print('---------------')
            
            return distances


    def load_organize_and_subsample_pointclouds(self, dataset='LFW', num_points=2048, verbose=True):
        # Load test dataset
        folds_pair_data, folds_pair_labels, folds_indexes = self.load_dataset(dataset_name=dataset, verbose=verbose)

        folds_pair_cache = self.organize_and_subsample_pointcloud(folds_pair_data, npoints=num_points, verbose=verbose)
        
        return folds_pair_cache, folds_pair_labels, folds_indexes


    def find_best_treshold_train_eval_test_set(self, train_distances, train_pair_labels, test_distances, test_pair_labels, verbose=True):
        if verbose:
            print('Findind best treshold...')

        best_train_tresh, best_train_acc, train_tar, train_far = self.find_best_treshold(train_distances, train_pair_labels, verbose=verbose)

        test_pair_labels = torch.tensor(test_pair_labels, dtype=torch.int8)
        test_tp, test_fp, test_tn, test_fn, test_acc, test_tar, test_far = self.eval_one_treshold(test_distances, test_pair_labels, best_train_tresh, verbose)

        return best_train_tresh, best_train_acc, train_tar, train_far, test_acc, test_tar, test_far, test_tp, test_fp, test_tn, test_fn


    def calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(
            np.logical_and(np.logical_not(predict_issame),
                        np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc


    # def calculate_roc(self, thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    def calculate_roc(self, thresholds, dist, actual_issame, nrof_folds=10, verbose=True):
        # assert (embeddings1.shape[0] == embeddings2.shape[0])
        # assert (embeddings1.shape[1] == embeddings2.shape[1])
        assert (dist.shape[0] == actual_issame.shape[0])   # Bernardo
        nrof_pairs = min(len(actual_issame), dist.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if verbose:
                print(f'calculate_roc - fold_idx: {fold_idx}/{nrof_folds-1}')

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):            
                _, _, acc_train[threshold_idx] = self.calculate_accuracy(
                    threshold, dist[train_set], actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = self.calculate_accuracy(
                    threshold, dist[test_set],
                    actual_issame[test_set])
            _, _, accuracy[fold_idx] = self.calculate_accuracy(
                thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        return tpr, fpr, accuracy


    def calculate_val_far(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(
            np.logical_and(predict_issame, np.logical_not(actual_issame)))
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))
        val = float(true_accept) / float(n_same)
        far = float(false_accept) / float(n_diff)
        return val, far


    # def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    def calculate_val(self, thresholds, dist, actual_issame, far_target, nrof_folds=10, verbose=True):
        # assert (embeddings1.shape[0] == embeddings2.shape[0])
        # assert (embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), dist.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        val = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)

        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff), 1)
        indices = np.arange(nrof_pairs)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if verbose:
                print(f'calculate_val - fold_idx: {fold_idx}/{nrof_folds-1}')

            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = self.calculate_val_far(
                    threshold, dist[train_set], actual_issame[train_set])
            if np.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0

            val[fold_idx], far[fold_idx] = self.calculate_val_far(
                threshold, dist[test_set], actual_issame[test_set])

        val_mean = np.mean(val)
        far_mean = np.mean(far)
        val_std = np.std(val)
        return val_mean, val_std, far_mean


    def do_k_fold_test(self, folds_pair_distances, folds_pair_labels, folds_indexes, verbose=True):
        thresholds = np.arange(0, 4, 0.01)
        tpr, fpr, accuracy = self.calculate_roc(thresholds, folds_pair_distances, folds_pair_labels, nrof_folds=10, verbose=verbose)

        if verbose:
            print('------------')

        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far = self.calculate_val(thresholds, folds_pair_distances, folds_pair_labels, far_target=1e-3, nrof_folds=10, verbose=verbose)

        if verbose:
            print('------------')

        return tpr, fpr, accuracy, val, val_std, far


    def do_verification_test(self, model, dataset='LFW', num_points=2048, verbose=True):
        model.eval()

        # # Load one point cloud (test)
        # path_point_cloud = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_reduced/m.0ql2bgg/0-FaceId-0/mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
        # points = load_one_point_cloud(path_point_cloud)

        folds_pair_cache, folds_pair_labels, folds_indexes = self.load_organize_and_subsample_pointclouds(dataset, num_points, verbose=verbose)

        folds_pair_distances = self.compute_set_distances(model, folds_pair_cache, verbose=verbose)
        folds_pair_distances = folds_pair_distances.cpu().detach().numpy()

        _, _, accuracy, val, val_std, far = self.do_k_fold_test(folds_pair_distances, folds_pair_labels, folds_indexes, verbose=verbose)
        acc_mean, acc_std = np.mean(accuracy), np.std(accuracy)
        # print(f'acc_mean={acc_mean},    acc_std={acc_std}')
        
        return acc_mean, acc_std



if __name__ == "__main__":

    verif_tester = VerificationTester()

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

    acc_mean, acc_std = verif_tester.do_verification_test(model, args.dataset, args.num_points, verbose=True)

    print('\nFinal - dataset: %s  -  acc_mean: %.6f    acc_std: %.6f)' % (args.dataset, acc_mean, acc_std))
    print('Finished!')
