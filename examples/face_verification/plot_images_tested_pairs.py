
import sys, os, numpy as np
from glob import glob
import shutil
import matplotlib.pyplot as plt
import argparse
import cv2

from dataloaders.tree_lfw import TreeLFW
from dataloaders.lfw_pairs_3Dreconstructed_MICA import LFW_Pairs_3DReconstructedMICA


def parse_args():
    parser = argparse.ArgumentParser('plot_images_tested_pairs.py')
    parser.add_argument('--results', type=str, required=False, help='log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230608-104723-F7xYY2Vnp4xS6EX28h3JxA/results/results_model=ResNet+PointNeXt_fusion=mean_checkpoint=_ckpt_best.pth_num_points=1024_dataset=lfw.npy', default='log/ms1mv3_3d_arcface/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6113-20230503-171328-SW2CTnmUDWBMoaVuSp4a4v/pointnext-s_arcface.yaml')
    parser.add_argument('--dataset', type=str, default='lfw', help='dataset name')
    parser.add_argument('--pairs_path', type=str, default='/nobackup1/bjgbiesseck/datasets/MICA_3Dreconstruction/lfw/pairs.txt', help='')
    parser.add_argument('--img_path', type=str, default='/nobackup1/bjgbiesseck/datasets/lfw', help='')
    parser.add_argument('--mesh_path', type=str, default='/nobackup1/bjgbiesseck/datasets/MICA_3Dreconstruction/lfw', help='')
    parser.add_argument('--fusion', type=str, default='mean', help='')

    args, opts = parser.parse_known_args()
    return args, opts


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


if __name__ == "__main__":
    args, opts = parse_args()

    if not os.path.isfile(args.pairs_path):
        print('\nError, no such file or directory:', args.pairs_path)
    if not os.path.isdir(args.img_path):
        print('\nError, no such file or directory:', args.img_path)
    if not os.path.isdir(args.mesh_path):
        print('\nError, no such file or directory:', args.mesh_path)
    
    results_dict = np.load(args.results, allow_pickle=True).item()
    # print('results_dict.keys():', results_dict.keys())

    all_img_pairs_paths_label, img_folds_indexes, _, _ = TreeLFW().load_img_pairs_from_protocol_file(args.img_path, args.pairs_path, '.jpg')
    # for i in range(len(all_img_pairs_paths_label)):
    #     print(f'all_img_pairs_paths_label[{i}]: {all_img_pairs_paths_label[i]}')

    all_mesh_pairs_paths_label, mesh_folds_indexes, _, _ = LFW_Pairs_3DReconstructedMICA().load_pointclouds_pairs_with_labels(args.mesh_path, args.pairs_path, '.jpg')
    # for i in range(len(all_mesh_pairs_paths_label)):
    #     print(f'all_mesh_pairs_paths_label[{i}]: {all_mesh_pairs_paths_label[i]}')

    assert len(all_img_pairs_paths_label) == len(all_mesh_pairs_paths_label)

    path_save_results = args.results.split('.npy')[0]
    # if os.path.isdir(path_save_results):
    #     print('\nDeleting previous results dir:', path_save_results)
    #     shutil.rmtree(path_save_results, ignore_errors=True)
    
    if not os.path.isdir(path_save_results):
        print('\nCreating results dir:', path_save_results)
        os.makedirs(path_save_results)
        print('')
    
    # sets_to_plot = ['fp_idx']
    sets_to_plot = ['tp_idx', 'fp_idx', 'tn_idx', 'fn_idx']
    
    # print('results_dict.keys():', results_dict.keys())
    # print('results_dict[\'mean\'].keys():', results_dict['mean'].keys())
    # print('results_dict[\'mean\'][\'tp_idx\']:', results_dict['mean']['tp_idx'])
    # print('len(results_dict[\'mean\'][\'tp_idx\']):', len(results_dict['mean']['tp_idx']))
    
    for set_to_plot in sets_to_plot:
        path_set_save_results = path_save_results + '/' + set_to_plot
        if os.path.isdir(path_set_save_results):
            print('\nDeleting previous results dir:', path_set_save_results)
            shutil.rmtree(path_set_save_results, ignore_errors=True)
        os.makedirs(path_set_save_results)

        results_to_plot = results_dict
        if 'mean' in list(results_dict.keys()):
            results_to_plot = results_dict['mean']

        for i, idx in enumerate(results_to_plot[set_to_plot]):
            print(set_to_plot + ' - ' + str(i) + '/' + str(len(results_to_plot[set_to_plot])), end='\r')
            img_pair_label, img_path0, img_path1 = all_img_pairs_paths_label[idx]
            mesh_pair_label, mesh_path0, mesh_path1 = all_mesh_pairs_paths_label[idx]
            
            assert int(img_pair_label) == int(mesh_pair_label)

            # img0 = cv2.imread(img_path0)
            # img1 = cv2.imread(img_path1)
            # mesh0 = cv2.imread(mesh_path0)
            # mesh1 = cv2.imread(mesh_path1)
            img0 = cv2.cvtColor(cv2.imread(img_path0), cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2RGB)
            mesh0 = cv2.cvtColor(cv2.imread(mesh_path0), cv2.COLOR_BGR2RGB)
            mesh1 = cv2.cvtColor(cv2.imread(mesh_path1), cv2.COLOR_BGR2RGB)
            
            path_pair_plot_save = path_set_save_results + '/' + set_to_plot + '_pair-idx_' + str(idx) + '.png'
            title = set_to_plot + ' - pair-idx: ' + str(idx) + ' - true-label: ' + str(img_pair_label)
            subtitle = img_path0.split('/')[-1] + '   ' + img_path1.split('/')[-1]
            print('')
            print('path_pair_plot_save:', path_pair_plot_save)
            print('title:', title)
            print('subtitle:', subtitle)
            save_pair_plot(path_pair_plot_save, title, 12, subtitle, 10, [img0, mesh0, img1, mesh1])
            
            # exit(0)
            print('--------')

            # img1, mesh0, mesh1 = load_images_pair(all_img_pairs_paths_label[idx], all_mesh_pairs_paths_label[idx])
        print('')

    print('\nFinished!')
    