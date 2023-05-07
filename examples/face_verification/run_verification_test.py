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

from .dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA



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
        # LFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairs.txt'
        self.LFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairsDevTest.txt'

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
            all_pairs_paths_label, pos_pair_label, neg_pair_label = LFW_Pairs_3DReconstructedMICA().load_pointclouds_pairs_with_labels(self.LFW_POINT_CLOUDS, self.LFW_VERIF_PAIRS_LIST, file_ext)
            
            if verbose:
                # print('\nLFW_Pairs_3DReconstructedMICA - load_pointclouds_pairs_with_labels')
                # print('all_pairs_paths_label:', all_pairs_paths_label)
                # print('len(all_pairs_paths_label):', len(all_pairs_paths_label))
                # print('pos_pair_label:', pos_pair_label)
                # print('neg_pair_label:', neg_pair_label)
                print('Loading dataset:', dataset_name)
        
        dataset = self.load_point_clouds_from_disk(all_pairs_paths_label, verbose=verbose)
        return dataset


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


    def compute_embeddings_distance(self, face_embedd, verbose=True):
        assert face_embedd.size()[0] % 2 == 0
        distances = torch.zeros(int(face_embedd.size()[0]/2))
        for i in range(0, face_embedd.size()[0], 2):
            embedd0, embedd1 = face_embedd[i], face_embedd[i+1]
            distances[int(i/2)] = torch.sum( torch.square( F.normalize(torch.unsqueeze(embedd0, 0)) - F.normalize(torch.unsqueeze(embedd1, 0)) ) )
        # if verbose:
        #     print('distances:', distances)
        #     print('distances.size():', distances.size())
        return distances


    def find_best_treshold(self, dataset, cos_sims, verbose=True):
        best_tresh = 0
        best_acc = 0
        
        # start, end, step = 0, 1, 0.01
        start, end, step = 0, 4, 0.01    # used in insightface code

        treshs = torch.arange(start, end+step, step)
        for i, tresh in enumerate(treshs):
            # torch.set_printoptions(precision=3)
            # tresh = torch.round(tresh, decimals=3)
            # tresh = round(tresh, 3)
            tp, fp, tn, fn, acc = 0, 0, 0, 0, 0
            for j, cos_sim in enumerate(cos_sims):
                _, _, pair_label = dataset[j]
                # print('pair_label:', pair_label)
                if pair_label == '1':  # positive pair
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
            # print(f'tester_multitask_FACEVERIFICATION - {i}/{treshs.size()[0]-1} - tresh: {tresh} - acc: {acc}')

            if acc > best_acc:
                best_acc = acc
                best_tresh = tresh

            if verbose:
                print('\x1b[2K', end='')
                print(f'tester_multitask_FACEVERIFICATION - {i}/{len(treshs)-1} - tresh: {tresh}', end='\r')

        return best_tresh, best_acc


    def do_verification_test(self, model, dataset='LFW', num_points=1200, verbose=True):
        model.eval()

        # # Load one point cloud (test)
        # path_point_cloud = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_reduced/m.0ql2bgg/0-FaceId-0/mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
        # points = load_one_point_cloud(path_point_cloud)

        # Load test dataset
        dataset = self.load_dataset(dataset_name=dataset, verbose=verbose)

        cache = self.organize_and_subsample_pointcloud(dataset, npoints=num_points, verbose=verbose)
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
                
                dist = self.compute_embeddings_distance(embedd, verbose=verbose)
                
                if verbose:
                    print('dist.size():', dist.size())
                
                distances[int(start_batch_idx/2):int(end_batch_idx/2)] = dist

                if verbose:
                    print('---------------')

            # print('distances:', distances)
            # print('distances.size():', distances.size())

            if verbose:
                print('Findind best treshold...')
            best_tresh, best_acc = self.find_best_treshold(dataset, distances, verbose=verbose)
            return best_tresh, best_acc



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

    best_tresh, best_acc = verif_tester.do_verification_test(model, args.dataset, args.num_points, verbose=True)
    print('\nFinal - best_tresh:', best_tresh, '    best_acc:', best_acc)

    print('Finished!')





'''
# BACKUP (WORKING)
if __name__ == "__main__":
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
    model, best_epoch, metrics = load_trained_weights_from_cfg_file(args.cfg)
    model.eval()

    # # Load one point cloud (test)
    # path_point_cloud = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_reduced/m.0ql2bgg/0-FaceId-0/mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
    # points = load_one_point_cloud(path_point_cloud)

    # Load test dataset
    dataset = load_dataset(dataset_name=args.dataset)

    cache = organize_and_subsample_pointcloud(dataset, npoints=args.num_points)
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
            # print('embedd:', embedd)
            print(f'computing face embeddings - batch_size: {batch_size} - batch {i}/{num_batches-1} - batch_idxs: {start_batch_idx}:{end_batch_idx} - embedd.size():', embedd.size())

            print('computing distances')
            dist = compute_embeddings_distance(embedd)
            print('dist.size():', dist.size())
            distances[int(start_batch_idx/2):int(end_batch_idx/2)] = dist

            print('---------------')

        # print('distances:', distances)
        # print('distances.size():', distances.size())

        print('Findind best treshold...')
        best_tresh, best_acc = find_best_treshold(dataset, distances)
        print('\nFinal - best_tresh:', best_tresh, '    best_acc:', best_acc)

        print('Finished!')
'''
        