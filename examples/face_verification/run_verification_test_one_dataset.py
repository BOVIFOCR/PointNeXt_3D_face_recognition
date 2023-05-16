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

try:
    from .dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA
except ImportError as e:
    from dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA



def parse_args():
    parser = argparse.ArgumentParser('run_verification_test.py')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--dataset', type=str, default='lfw', help='dataset name')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points to subsample')
    # parser.add_argument('--pretrained', type=str, required=True, help='checkpoint_file.pth')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    return args, opts


class VerificationTester:
    
    def __init__(self):
        self.LFW_POINT_CLOUDS = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw'
        self.LFW_TRAIN_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairs.txt'        # whole dataset (6000 face pairs)
        self.LFW_TEST_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairsDevTest.txt'  # only test set (1000 face pairs)

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


    def load_trained_weights_from_cfg_file(self, cfg_path='log/ms1mv2_3d_arcface/ms1mv2_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6520-20230429-170750-98s4kHjBdbgbRWcegn9V9y/pointnext-s_arcface.yaml'):
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
            all_train_pairs_paths_label, pos_pair_label, neg_pair_label = LFW_Pairs_3DReconstructedMICA().load_pointclouds_pairs_with_labels(self.LFW_POINT_CLOUDS, self.LFW_TRAIN_VERIF_PAIRS_LIST, file_ext)
            all_test_pairs_paths_label, pos_pair_label, neg_pair_label = LFW_Pairs_3DReconstructedMICA().load_pointclouds_pairs_with_labels(self.LFW_POINT_CLOUDS, self.LFW_TEST_VERIF_PAIRS_LIST, file_ext)

            if verbose:
                # print('\nLFW_Pairs_3DReconstructedMICA - load_pointclouds_pairs_with_labels')
                # print('all_pairs_paths_label:', all_pairs_paths_label)
                # print('len(all_pairs_paths_label):', len(all_pairs_paths_label))
                # print('pos_pair_label:', pos_pair_label)
                # print('neg_pair_label:', neg_pair_label)
                print('Loading dataset:', dataset_name)
        
        train_dataset = self.load_point_clouds_from_disk(all_train_pairs_paths_label, verbose=verbose)
        test_dataset = self.load_point_clouds_from_disk(all_test_pairs_paths_label, verbose=verbose)
        return train_dataset, test_dataset


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


    def eval_one_treshold(self, cos_sims, pair_labels, tresh=2.0, verbose=True):
        tp, fp, tn, fn = 0., 0., 0., 0.
        for j, (cos_sim, pair_label) in enumerate(zip(cos_sims, pair_labels)):
            if pair_label == 1:   # positive pair
                if cos_sim < tresh:
                    tp += 1
                else:
                    fn += 1
            else:  # negative pair
                if cos_sim >= tresh:
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


    def find_best_treshold(self, dataset, cos_sims, verbose=True):
        best_tresh = 0
        best_acc = 0
        
        # start, end, step = 0, 1, 0.01   # used in insightface code
        start, end, step = 0, 2, 0.005
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

        pair_labels = torch.tensor([int(dataset[j][2]) for j in range(len(dataset))], dtype=torch.int8)   # dataset[j] is (pc0, pc1, pair_label)

        treshs = torch.arange(start, end+step, step)
        for i, tresh in enumerate(treshs):

            all_tp_eval[i], all_fp_eval[i], all_tn_eval[i], all_fn_eval[i], all_acc_eval[i], all_tar_eval[i], all_far_eval[i] = self.eval_one_treshold(cos_sims, pair_labels, tresh, verbose)

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


    def compute_set_distances(self, cache={}, verbose=True):
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


    def do_verification_test(self, model, dataset='LFW', num_points=1200, verbose=True):
        model.eval()

        # # Load one point cloud (test)
        # path_point_cloud = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_reduced/m.0ql2bgg/0-FaceId-0/mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
        # points = load_one_point_cloud(path_point_cloud)

        # Load test dataset
        train_dataset, test_dataset = self.load_dataset(dataset_name=dataset, verbose=verbose)

        train_cache = self.organize_and_subsample_pointcloud(train_dataset, npoints=num_points, verbose=verbose)
        test_cache = self.organize_and_subsample_pointcloud(test_dataset, npoints=num_points, verbose=verbose)

        train_distances = self.compute_set_distances(train_cache, verbose=verbose)
        test_distances = self.compute_set_distances(test_cache, verbose=verbose)

        # print('distances:', distances)
        # print('distances.size():', distances.size())
        if verbose:
            print('Findind best treshold...')

        best_train_tresh, best_train_acc, train_tar, train_far = self.find_best_treshold(train_dataset, train_distances, verbose=verbose)

        test_pair_labels = torch.tensor([int(test_dataset[j][2]) for j in range(len(test_dataset))], dtype=torch.int8)   # dataset[j] is (pc0, pc1, pair_label)
        test_tp, test_fp, test_tn, test_fn, test_acc, test_tar, test_far = self.eval_one_treshold(test_distances, test_pair_labels, best_train_tresh, verbose)

        return best_train_tresh, best_train_acc, train_tar, train_far, \
                test_acc, test_tar, test_far




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
    model, best_epoch, metrics = verif_tester.load_trained_weights_from_cfg_file(args.cfg)
    model.eval()

    best_train_tresh, best_train_acc, train_tar, train_far, \
    test_acc, test_tar, test_far = verif_tester.do_verification_test(model, args.dataset, args.num_points, verbose=True)
    
    print('\nFinal - dataset: %s  -  (best_train_tresh: %.6f    best_train_acc: %.6f)    (train_tar: %.6f    train_far: %.10f)' % (args.dataset, best_train_tresh, best_train_acc, train_tar, train_far))
    print('                         (test_acc: %.6f    test_tar: %.6f    test_far: %.10f)' % (test_acc, test_tar, test_far))
    print('Finished!')
