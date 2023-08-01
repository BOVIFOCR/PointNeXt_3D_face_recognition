# coding: utf-8
# Source: https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval_ijbc.py
import __init__
import sys, os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
# from backbones import get_model
from sklearn.metrics import roc_curve, auc

from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path

import sys
import warnings

# import tensorflow as tf
import yaml
from openpoints.utils import EasyConfig
from run_verification_test_one_dataset import VerificationTester
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
# from model import get_embd


sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')
# general
# parser.add_argument('--config-path', default='./configs/config_res50_ms1mv2-1000subj.yaml', help='path to load model.')
parser.add_argument('--config-path', default='log/ms1mv3_3d_arcface/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6113-20230503-171328-SW2CTnmUDWBMoaVuSp4a4v/pointnext-s_arcface.yaml', help='path to load model.')
# parser.add_argument('--model-prefix', default='output/dataset=MS1MV3_1000subj_classes=1000_backbone=resnet-v2-m-50_epoch-num=100_margin=0.5_scale=64.0_lr=0.01_wd=0.0005_momentum=0.9_20230518-004011/checkpoints/ckpt-m-100000', help='path to load model.')
parser.add_argument('--image-path', default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/IJB-C/IJB/IJB-C/rec_data_ijbc', type=str, help='')
parser.add_argument('--result-dir', default='results_ijbc_template', type=str, help='')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--network', default='iresnet50', type=str, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--num_points', type=int, default=1024, help='number of points to subsample')
# args = parser.parse_args()
args, opts = parser.parse_known_args()

target = args.target
config_path = args.config_path
# model_path = args.model_prefix
model_path = '/'.join(args.config_path.split('/')[:-1])
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
use_norm_score = True  # if Ture, TestMode(N1)
# use_detector_score = True  # if Ture, TestMode(D1)   # original (for 2D images)
use_detector_score = False   # if Ture, TestMode(D1)   # Bernardo (for 3D point clouds)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size

NUM_POINTS = args.num_points
protocols_path = '/datasets1/bjgbiesseck/IJB-C/rec_data_ijbc'


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size

        # original
        # weight = torch.load(prefix)
        # resnet = get_model(args.network, dropout=0, fp16=False).cuda()
        # resnet.load_state_dict(weight)
        # model = torch.nn.DataParallel(resnet)
        # self.model = model
        # self.model.eval()

        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        # img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        # img_flip = np.transpose(img_flip, (2, 0, 1))
        # input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob = np.zeros((2, self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

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
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = torch.gather(points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        return points

    # Bernardo
    def organize_and_subsample_pointcloud(self, dataset, npoints=1024, verbose=True):
        chanels = 3
        cache = {}
        # cache['pos'] = torch.zeros(len(dataset), chanels, npoints)
        cache['x'] = torch.zeros(size=(len(dataset), npoints, chanels), device=0)
        for i, pair in enumerate(dataset):
            # pc0, pc1, pair_label = pair
            pc0 = pair
            pc0_orig_shape = pc0.shape
            pc0 = self.subsample_point_cloud(pc0[:, :3], npoints)
            # pc1 = self.subsample_point_cloud(pc1[:, :3], npoints)
            # j = i * 2
            cache['x'][i] = pc0
            # cache['x'][j+1] = pc1

            if verbose:
                print(f'organize_and_subsample_pointcloud - pair {i}/{len(dataset)-1} - pc0: {pc0_orig_shape} ->', pc0.size())
        # print('cache[\'x\'].size():', cache['x'].size())
        
        cache['pos'] = cache['x'].contiguous()
        cache['x'] = cache['x'].transpose(1, 2).contiguous()
        return cache

    @torch.no_grad()
    def forward_db(self, cache, model):
        # imgs = torch.Tensor(batch_data).cuda()
        # imgs = torch.Tensor(batch_data)
        # imgs.div_(255).sub_(0.5).div_(0.5)
        
        # feat = self.model(imgs)
        data = {}
        data['pos'] = cache['pos']
        data['x']   = cache['x']
        embedd = model.get_face_embedding(data)
        
        feat = embedd

        # print('embedd:', embedd)
        # print('embedd.shape:', embedd.shape)
        # sys.exit(0)
        
        # feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        # feat = feat.reshape([self.batch_size, feat.shape[1]])
        return feat.cpu().numpy()
        # return feat


# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


# In[ ]:


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# In[ ]:


def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    batch_size = args.batch_size
    # data_shape = (3, 112, 112)   # original
    # data_shape = (112, 112, 3)   # Bernardo
    data_shape = (2900, 3)         # Bernardo

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    # img_feats = np.empty((len(files), 1024), dtype=np.float32)
    img_feats = np.empty((len(files), 512), dtype=np.float32)

    # LOAD MODEL
    verif_tester = VerificationTester()
    cfg = EasyConfig()
    cfg.load(config_path, recursive=True)
    cfg.update(opts)
    if not hasattr(cfg, 'seed'):
        cfg.seed = np.random.randint(1, 10000)
    print('Building model...')
    model = build_model_from_cfg(cfg.model).to(0)
    # Load trained weights
    model, best_epoch, metrics = verif_tester.load_trained_weights_from_cfg_file(model, config_path, '_ckpt_best.pth')
    model.eval()
    
    # COMPUTE EMBEDDINGS
    with torch.no_grad():
        # batch_data = np.empty((2 * batch_size, 3, 112, 112))
        # batch_data = np.empty((2 * batch_size, 112, 112, 3))
        batch_data = np.empty((batch_size, 2900, 3))
        embedding = Embedding(model_path, data_shape, batch_size)
        for img_index, each_line in enumerate(files[:len(files) - rare_size]):
            name_lmk_score = each_line.strip().split(' ')
            # img_name = os.path.join(img_path, name_lmk_score[0])
            # img = cv2.imread(img_name)
            # lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
            # lmk = lmk.reshape((5, 2))
            # input_blob = embedding.get(img, lmk)

            file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'  # Bernardo
            folder_name = name_lmk_score[0].split('.')[0]                             # Bernardo
            img_name = os.path.join(img_path, folder_name, file_ext)                  # Bernardo

            # print('get_image_feature - img_name:', img_name)

            # Bernardo
            if img_name.endswith('.npy'):
                point_set = np.load(img_name).astype(np.float32)
            
            if point_set.shape[1] > 3:        # if contains normals and curvature
                point_set = point_set[:,0:3]  # remove normals and curvature
            point_set = verif_tester.pc_normalize(point_set)

            # batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
            # batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
            batch_data[(img_index - batch * batch_size)][:] = point_set[:2900]
            if (img_index + 1) % batch_size == 0:
                print('batch', batch)
                folds_pair_cache = embedding.organize_and_subsample_pointcloud(batch_data, npoints=NUM_POINTS, verbose=False)
                img_feats[batch * batch_size:batch * batch_size +
                                            batch_size][:] = embedding.forward_db(folds_pair_cache, model)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])

        # batch_data = np.empty((2 * rare_size, 3, 112, 112))   # original
        # batch_data = np.empty((2 * rare_size, 112, 112, 3))   # Bernardo
        batch_data = np.empty((rare_size, 2900, 3))            # Bernardo
        embedding = Embedding(model_path, data_shape, rare_size)
        for img_index, each_line in enumerate(files[len(files) - rare_size:]):
            name_lmk_score = each_line.strip().split(' ')
            # img_name = os.path.join(img_path, name_lmk_score[0])
            # img = cv2.imread(img_name)
            # lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
            # lmk = lmk.reshape((5, 2))
            # input_blob = embedding.get(img, lmk)

            file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'  # Bernardo
            folder_name = name_lmk_score[0].split('.')[0]                             # Bernardo
            img_name = os.path.join(img_path, folder_name, file_ext)                  # Bernardo

            # Bernardo
            if img_name.endswith('.npy'):
                point_set = np.load(img_name).astype(np.float32)
            
            if point_set.shape[1] > 3:        # if contains normals and curvature
                point_set = point_set[:,0:3]  # remove normals and curvature
            point_set = verif_tester.pc_normalize(point_set)

            # batch_data[2 * img_index][:] = input_blob[0]
            # batch_data[2 * img_index + 1][:] = input_blob[1]
            batch_data[img_index][:] = point_set[:2900]
            if (img_index + 1) % rare_size == 0:
                print('batch', batch)
                folds_pair_cache = embedding.organize_and_subsample_pointcloud(batch_data, npoints=NUM_POINTS, verbose=False)
                img_feats[len(files) -
                        rare_size:][:] = embedding.forward_db(folds_pair_cache, model)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
        # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
        return img_feats, faceness_scores


# In[ ]:


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


# In[ ]:


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats



exper_id = model_path.split('/')[-1]            # Bernardo
save_path = os.path.join(result_dir, exper_id)  # Bernardo
score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
label_save_file = os.path.join(save_path, "label.npy")
img_feats_save_file = os.path.join(save_path, "img_feats.npy")
faceness_scores_save_file = os.path.join(save_path, "faceness_scores.npy")



# # Step1: Load Meta Data

# In[ ]:

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    # os.path.join('%s/meta' % image_path, '%s_face_tid_mid.txt' % target.lower()))      # original
    os.path.join('%s/meta' % protocols_path, '%s_face_tid_mid.txt' % target.lower()))    # Bernardo
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    # os.path.join('%s/meta' % image_path, '%s_template_pair_label.txt' % target.lower()))     # original
    os.path.join('%s/meta' % protocols_path, '%s_template_pair_label.txt' % target.lower()))   # Bernardo
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))



# Bernardo
if not os.path.exists(img_feats_save_file):
    # # Step 2: Get Image Features

    # In[ ]:

    # =============================================================
    # load image features
    # format:
    #           img_feats: [image_num x feats_dim] (227630, 512)
    # =============================================================
    start = timeit.default_timer()
    # img_path = '%s/loose_crop' % image_path
    img_path = '%s/refined_img' % image_path
    # img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())     # original
    img_list_path = '%s/meta/%s_name_5pts_score.txt' % (protocols_path, target.lower())   # Bernardo
    img_list = open(img_list_path)
    files = img_list.readlines()
    # files_list = divideIntoNstrand(files, rank_size)
    files_list = files

    # img_feats
    # for i in range(rank_size):
    img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                                model_path, 0, gpu_id)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                            img_feats.shape[1]))


    # Bernardo
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
    print('Saving img_feats:', img_feats_save_file)
    np.save(img_feats_save_file, img_feats)
    print('Saving faceness_scores:', img_feats_save_file)
    np.save(faceness_scores_save_file, faceness_scores)

else:
    print('Loading img_feats:', img_feats_save_file)
    img_feats = np.load(img_feats_save_file)
    print('Loading faceness_scores:', img_feats_save_file)
    faceness_scores = np.load(faceness_scores_save_file)



# # Step3: Get Template Features

# In[ ]:

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                    2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))



# # Step 4: Get Template Similarity Scores

# In[ ]:

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))


# In[ ]:
# exper_id = model_path.split('/')[-2]            # Bernardo
# save_path = os.path.join(result_dir, exper_id)  # Bernardo
# save_path = os.path.join(result_dir, args.job)
# save_path = result_dir + '/%s_result' % target

if not os.path.exists(save_path):
    os.makedirs(save_path)
# score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
print('Saving scores:', score_save_file)
np.save(score_save_file, score)
print('Saving labels:', label_save_file)
np.save(label_save_file, label)





# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
roc_auc = 0.0
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
print(tpr_fpr_table)

# Bernardo
print('AUC = %0.4f %%' % (roc_auc * 100))
