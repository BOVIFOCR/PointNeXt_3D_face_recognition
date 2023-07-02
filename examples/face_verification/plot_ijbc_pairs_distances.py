
import sys, os, numpy as np
from glob import glob
import shutil
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser('plot_images_tested_pairs.py')
    parser.add_argument('--dist', type=str, required=False, help='', default='/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed2501-20230504-104157-7x3QoJexVmWZvcgzWuH2wV/ijbc.npy')
    parser.add_argument('--label', type=str, required=False, help='', default='/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed2501-20230504-104157-7x3QoJexVmWZvcgzWuH2wV/label.npy')
    parser.add_argument('--num_pairs', type=int, required=False, help='', default=5000)
    parser.add_argument('--results', type=str, required=False, help='', default='plots')
    parser.add_argument('--title', type=str, required=False, help='', default='PointNeXt - Cosine distances between pairs - IJB-C')
    parser.add_argument('--subtitle', type=str, required=False, help='', default='Train: MS1MV3 (10k classes)')

    args, opts = parser.parse_known_args()
    return args, opts


# def save_pair_plot(path_to_save, title, title_size, subtitle, subtitle_size, images):
def plot_distances_between_pairs(dists, labels, title, subtitle, path_save_plot='', save=True, show=False):
    fig, ax = plt.subplots(2, 1)

    fig.suptitle(title, fontsize=12, wrap=True)
    # plt.title(subtitle, fontsize=10, wrap=True)
    # fig.text(0.5, 0.05, subtitle, fontsize=subtitle_size, ha='center')
    # fig.text(0.5, 0.9, subtitle, fontsize=subtitle_size, ha='center')

    ax[0].plot(np.arange(0, len(dists), dtype=int), dists)
    ax[0].set(xlabel='Pair index', ylabel='Cosine distance')
    ax[0].set_title(subtitle)
    
    ax[1].scatter(np.arange(0, len(dists), dtype=int), labels)
    ax[1].set(xlabel='Pair index', ylabel='Pair label\n(1=genuine, 0=impostor)')

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    
    if show:
        plt.show()

    if save:
        print('Saving figure:', path_save_plot)
        plt.savefig(path_save_plot)
    plt.close()


def sample_pairs(dists, labels, num_pairs, method='sequential'):
    if method == 'sequential':
        idx_genuine_pairs = np.where(labels == 1)[0][:num_pairs//2]
        idx_impostor_pairs = np.where(labels == 0)[0][:num_pairs//2]
    
    dists = np.concatenate((dists[idx_genuine_pairs], dists[idx_impostor_pairs]))
    labels = np.concatenate((labels[idx_genuine_pairs], labels[idx_impostor_pairs]))
    return dists, labels

if __name__ == "__main__":
    args, opts = parse_args()

    if not os.path.isfile(args.dist):
        print('\nError, no such file or directory:', args.dist)
        sys.exit(0)
    if not os.path.isfile(args.label):
        print('\nError, no such file or directory:', args.label)
        sys.exit(0)
    
    print('\nLoading dists:', args.dist)
    dists = np.load(args.dist)
    print('Loading labels:', args.dist)
    labels = np.load(args.label)
    assert len(dists) == len(labels)

    sample_dists, sample_labels = sample_pairs(dists, labels, args.num_pairs, method='sequential')

    path_save_results = '/'.join(args.dist.split('/')[:-1]) + '/' + args.results
    if not os.path.isdir(path_save_results):
        print('\nCreating results dir:', path_save_results)
        os.makedirs(path_save_results)
        print('')
    
    filename_plot = 'distances_num-pairs=' + str(args.num_pairs)
    path_save_plot = path_save_results + '/' + filename_plot
    # title = 'PointNeXt - Cosine distances between pairs - IJB-C'
    title = args.title
    subtitle = args.subtitle
    plot_distances_between_pairs(sample_dists, sample_labels, title, subtitle, path_save_plot, save=True, show=False)


    print('\nFinished!')
    