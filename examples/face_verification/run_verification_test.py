import __init__
import argparse, yaml
from torch import multiprocessing as mp

import sys, os, glob, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, cal_model_parm_nums, Wandb
from openpoints.utils import EasyConfig

from dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA



LFW_POINT_CLOUDS = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw'
LFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairs.txt'
# LFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairsDevTest.txt'

MLFW_POINT_CLOUDS = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MLFW'
# MLFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/MLFW/pairs.txt'


def parse_args():
    parser = argparse.ArgumentParser('run_verification_test.py')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--dataset', type=str, default='lfw', help='dataset name')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points to subsample')
    # parser.add_argument('--pretrained', type=str, required=True, help='checkpoint_file.pth')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    return args, opts


def pc_normalize(pc):
    pc /= 100
    pc = (pc - pc.min()) / (pc.max() - pc.min())
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def load_trained_weights_from_cfg_file(cfg_path='log/ms1mv2_3d_arcface/ms1mv2_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6520-20230429-170750-98s4kHjBdbgbRWcegn9V9y/pointnext-s_arcface.yaml'):
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


def load_one_point_cloud(file_path):
    if file_path.endswith('.npy'):
        points = np.load(file_path).astype(np.float32)
    points[:, :3] = pc_normalize(points[:, :3])
    return points


def load_point_clouds_from_disk(pairs_paths):
    data = [None] * len(pairs_paths)
    for i, pair in enumerate(pairs_paths):
        pair_label, path0, path1 = pair
        # print(f'pair: {i}/{len(pairs_paths)-1}', end='\r')
        print(f'pair: {i}/{len(pairs_paths)-1}')
        print('pair_label:', pair_label)
        print('path0:', path0)
        print('path1:', path1)
        pc0 = np.load(path0)
        pc1 = np.load(path1)
        # data[i] = (label, pc0, pc1)
        data[i] = (pc0, pc1, pair_label)
        print('------------')
    print()
    return data


def load_dataset(dataset_name='lfw'):
    if dataset_name.upper() == 'LFW':
        file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
        all_pairs_paths_label, pos_pair_label, neg_pair_label = LFW_Pairs_3DReconstructedMICA().load_pointclouds_pairs_with_labels(LFW_POINT_CLOUDS, LFW_VERIF_PAIRS_LIST, file_ext)
        # print('\nLFW_Pairs_3DReconstructedMICA - load_pointclouds_pairs_with_labels')
        # print('all_pairs_paths_label:', all_pairs_paths_label)
        # print('len(all_pairs_paths_label):', len(all_pairs_paths_label))
        # print('pos_pair_label:', pos_pair_label)
        # print('neg_pair_label:', neg_pair_label)
        print('Loading dataset:', dataset_name)
    
    data = load_point_clouds_from_disk(all_pairs_paths_label)
    return data


def subsample_point_cloud(points, npoints=1024):
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


def organize_and_subsample_pointcloud(data, npoints=1024):
    chanels = 3
    cache = {}
    # cache['pos'] = torch.zeros(len(data), chanels, npoints)
    cache['x'] = torch.zeros(size=(2*len(data), npoints, chanels), device=0)
    for i, pair in enumerate(data):
        pc0, pc1, pair_label = pair
        pc0_orig_shape, pc1_orig_shape = pc0.shape, pc1.shape
        pc0 = subsample_point_cloud(pc0[:, :3], npoints)
        pc1 = subsample_point_cloud(pc1[:, :3], npoints)
        j = i * 2
        cache['x'][j]   = pc0
        cache['x'][j+1] = pc1
        print(f'Pair {i}/{len(data)-1} - subsampling  pc0: {pc0_orig_shape} ->', pc0.size(), f',  pc1: {pc1_orig_shape} ->', pc1.size())
    # print('cache[\'x\'].size():', cache['x'].size())
    
    cache['pos'] = cache['x'].contiguous()
    cache['x'] = cache['x'].transpose(1, 2).contiguous()
    return cache


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
    data = load_dataset(dataset_name=args.dataset)
    
    cache = organize_and_subsample_pointcloud(data, npoints=args.num_points)
    print('cache[\'x\'].size():', cache['x'].size())

    with torch.no_grad():
        batch_size = 256
        num_batches = len(cache['x']) // batch_size
        last_batch_size = len(cache['x']) % batch_size
        if last_batch_size > 0: num_batches += 1
        for i in range(0, num_batches):
            j = i*batch_size
            num_samples = batch_size
            if j + batch_size > len(cache['x']): num_samples = last_batch_size

            data = {}
            data['pos'] = cache['pos'][j:j+num_samples]
            data['x']   = cache['x'][j:j+num_samples]
            embedd = model.get_face_embedding(data)
            # print('embedd:', embedd)
            print(f'batch {i}/{num_batches-1} - j: {j}:{j+num_samples} - embedd.size():', embedd.size())
