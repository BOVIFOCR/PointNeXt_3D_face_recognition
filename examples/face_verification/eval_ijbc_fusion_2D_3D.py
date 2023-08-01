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
parser.add_argument('--dist1', default='/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc/dataset=MS1MV3_1000subj_classes=1000_backbone=resnet-v2-m-50_epoch-num=100_margin=0.5_scale=64.0_lr=0.01_wd=0.0005_momentum=0.9_20230518-004011/ijbc.npy', help='path to load model.')
parser.add_argument('--dist2', default='/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc/dataset=reconst_mica_ms1mv2_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_06052023_114705/ijbc.npy', help='path to load model.')
parser.add_argument('--image-path', default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/IJB-C/IJB/IJB-C/rec_data_ijbc', type=str, help='')
parser.add_argument('--result-dir', default='results_ijbc', type=str, help='')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--network', default='iresnet50', type=str, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
# parser.add_argument('--num_points', type=int, default=1024, help='number of points to subsample')
# args = parser.parse_args()
args, opts = parser.parse_known_args()

target = args.target
# config_path = args.config_path
# model_path = args.model_prefix
# model_path = '/'.join(args.config_path.split('/')[:-1])
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
job = args.job
batch_size = args.batch_size

assert target == 'IJBC' or target == 'IJBB'



def fuse_scores(scores1, scores2):
    scores1 /= np.max(scores1)
    scores2 /= np.max(scores2)
    final_scores = (scores1 + scores2) / 2
    return final_scores



save_path = '/'.join(args.dist2.split('/')[:-1])
score_save_file1 = args.dist1
label_save_file1 = '/'.join(score_save_file1.split('/')[:-1]) + '/' + 'label.npy'
score_save_file2 = args.dist2
label_save_file2 = '/'.join(score_save_file2.split('/')[:-1]) + '/' + 'label.npy'

score_save_file = '/'.join(score_save_file2.split('/')[:-1]) + '/' + 'fused_scores.npy'



# Bernardo
print('\nLoading scores1:', score_save_file1)
score1 = np.load(score_save_file1)
print('Loading labels1:', label_save_file1)
label1 = np.load(label_save_file1)

print('\nLoading scores2:', score_save_file2)
score2 = np.load(score_save_file2)
print('Loading labels2:', label_save_file2)
label2 = np.load(label_save_file2)

assert np.all(label1 == label2)

print('\nFusing scores...')
fused_score = fuse_scores(score1, score2)
print('Saving fused_score:', score_save_file)
np.save(score_save_file, fused_score)





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
    # fpr, tpr, _ = roc_curve(label, scores[method])  # original
    fpr, tpr, _ = roc_curve(label1, scores[method])   # Bernardo
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
fig.savefig(os.path.join(save_path, '%s.pdf' % 'fused'.lower()))

path_save_tpr_fpr_table = os.path.join(save_path, '%s_tpr_fpr_table_fused.txt' % target.lower())
with open(path_save_tpr_fpr_table, 'w') as f:
    print(f'Saving tpr_fpr_table \'{path_save_tpr_fpr_table}\'')
    f.write('tpr_fpr_table\n')
    f.write(tpr_fpr_table.get_string())
    f.write('\nROC-AUC = %0.4f %%\n' % (roc_auc * 100))

print(tpr_fpr_table)
print('ROC-AUC = %0.4f %%' % (roc_auc * 100))
