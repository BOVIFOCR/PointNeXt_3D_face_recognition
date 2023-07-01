
import sys, os, numpy as np
from glob import glob
import shutil
import matplotlib.pyplot as plt
import argparse

from dataloaders.tree_lfw import TreeLFW
from dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA


def parse_args():
    parser = argparse.ArgumentParser('plot_images_tested_pairs.py')
    parser.add_argument('--results2d',   type=str, required=False, default='log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230608-104723-F7xYY2Vnp4xS6EX28h3JxA/results/results_model=ResNet_checkpoint=_ckpt_best.pth_num_points=2048_dataset=lfw.npy', help='', )
    parser.add_argument('--results3d',   type=str, required=False, default='log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230608-104723-F7xYY2Vnp4xS6EX28h3JxA/results/results_model=PointNeXt_checkpoint=_ckpt_best.pth_num_points=1024_dataset=lfw.npy', help='', )
    parser.add_argument('--results2d3d', type=str, required=False, default='log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230608-104723-F7xYY2Vnp4xS6EX28h3JxA/results/results_model=ResNet+PointNeXt_fusion=mean_checkpoint=_ckpt_best.pth_num_points=1024_dataset=lfw.npy', help='', )
    
    args, opts = parser.parse_known_args()
    return args, opts

'''
def save_pair_plot(path_to_save, title, title_size, subtitle, subtitle_size, images):
    fig, ax = plt.subplots(2, 2)

    fig.suptitle(title, fontsize=title_size)
    # fig.text(0.5, 0.05, subtitle, fontsize=subtitle_size, ha='center')
    fig.text(0.5, 0.9, subtitle, fontsize=subtitle_size, ha='center')

    for i in range(2):
        for j in range(2):
            ax[i, j].imshow(images[i * 2 + j])
            ax[i, j].axis("off")
    
    plt.savefig(path_to_save)
    plt.close()
'''


if __name__ == "__main__":
    args, opts = parse_args()

    if not os.path.isfile(args.results2d):
        print('\nError, no such file or directory:', args.results2d)
    if not os.path.isfile(args.results3d):
        print('\nError, no such file or directory:', args.results3d)
    if not os.path.isfile(args.results2d3d):
        print('\nError, no such file or directory:', args.results2d3d)
    
    # print('\nLoading results2d:', args.results2d)
    results2d_dict = np.load(args.results2d, allow_pickle=True).item()
    # print('\nLoading results3d:', args.results3d)
    results3d_dict = np.load(args.results3d, allow_pickle=True).item()
    # print('\nLoading results2d3d:', args.results2d3d)
    results2d3d_dict = np.load(args.results2d3d, allow_pickle=True).item()
    # print('results_dict.keys():', results_dict.keys())


    fusion_method = ['mean']
    samples = ['tp_idx', 'fp_idx', 'tn_idx', 'fn_idx']

    for fm in fusion_method:
        # print(f'results2d_dict.keys():', results2d_dict.keys())
        # print(f'results2d_dict[{fm}][\'tp_idx\']:', results2d_dict[fm]['tp_idx'])
        # print(f'results2d_dict[{fm}][\'fn_idx\']:', results2d_dict[fm]['fn_idx'])
        # print(f'results2d_dict[{fm}][\'tn_idx\']:', results2d_dict[fm]['tn_idx'])
        # print(f'results2d_dict[{fm}][\'fp_idx\']:', results2d_dict[fm]['fp_idx'])
        results2d_tp_idx = results2d_dict[fm]['tp_idx']
        results2d_fn_idx = results2d_dict[fm]['fn_idx']
        results2d_tn_idx = results2d_dict[fm]['tn_idx']
        results2d_fp_idx = results2d_dict[fm]['fp_idx']
        print('results2d_tp_idx.shape:', results2d_tp_idx.shape)
        print('results2d_fn_idx.shape:', results2d_fn_idx.shape)
        print('results2d_tn_idx.shape:', results2d_tn_idx.shape)
        print('results2d_fp_idx.shape:', results2d_fp_idx.shape)
        print()

        # print(f'results3d_dict.keys():', results3d_dict.keys())
        results3d_tp_idx = results3d_dict['tp_idx']
        results3d_fn_idx = results3d_dict['fn_idx']
        results3d_tn_idx = results3d_dict['tn_idx']
        results3d_fp_idx = results3d_dict['fp_idx']
        print('results3d_tp_idx.shape:', results3d_tp_idx.shape)
        print('results3d_fn_idx.shape:', results3d_fn_idx.shape)
        print('results3d_tn_idx.shape:', results3d_tn_idx.shape)
        print('results3d_fp_idx.shape:', results3d_fp_idx.shape)
        print()

        unique2d3d_tp_idx = np.unique(np.append(results2d_tp_idx, results3d_tp_idx))
        unique2d3d_fn_idx = np.unique(np.append(results2d_fn_idx, results3d_fn_idx))
        unique2d3d_tn_idx = np.unique(np.append(results2d_tn_idx, results3d_tn_idx))
        unique2d3d_fp_idx = np.unique(np.append(results2d_fp_idx, results3d_fp_idx))
        print('unique2d3d_tp_idx.shape:', unique2d3d_tp_idx.shape)
        print('unique2d3d_fn_idx.shape:', unique2d3d_fn_idx.shape)
        print('unique2d3d_tn_idx.shape:', unique2d3d_tn_idx.shape)
        print('unique2d3d_fp_idx.shape:', unique2d3d_fp_idx.shape)
        print()

        print('--------------------')

        # print(f'results2d3d_dict[{fm}].keys():', results2d3d_dict[fm].keys())
        results2d3d_tp_idx = results2d3d_dict[fm]['tp_idx']
        results2d3d_fn_idx = results2d3d_dict[fm]['fn_idx']
        results2d3d_tn_idx = results2d3d_dict[fm]['tn_idx']
        results2d3d_fp_idx = results2d3d_dict[fm]['fp_idx']
        print('results2d3d_tp_idx.shape:', results2d3d_tp_idx.shape)
        print('results2d3d_fn_idx.shape:', results2d3d_fn_idx.shape)
        print('results2d3d_tn_idx.shape:', results2d3d_tn_idx.shape)
        print('results2d3d_fp_idx.shape:', results2d3d_fp_idx.shape)
        print()

    print('\nFinished!')
    