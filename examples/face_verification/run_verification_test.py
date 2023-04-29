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
    
    # Load one point cloud (test)
    path_point_cloud = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_reduced/m.0ql2bgg/0-FaceId-0/mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
    points = np.load(path_point_cloud).astype(np.float32)
    points[:, :3] = pc_normalize(points[:, :3])
        
    points = torch.from_numpy(points).float().to(0)
    points = torch.unsqueeze(points, dim = 0)
    # print('points:', points)
    # print('points.shape:', points.shape)

    # Copied from 'train_one_epoch()' method
    num_curr_pts = points.shape[1]
    npoints = args.num_points
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

    data = {}
    data['pos'] = points[:, :, :3].contiguous()
    data['x'] = points[:, :, :3].transpose(1, 2).contiguous()

    with torch.no_grad():
        # logits = model(data)
        logits = model.get_face_embedding(data)
        # print('logits:', logits)
        print('logits.size():', logits.size())
