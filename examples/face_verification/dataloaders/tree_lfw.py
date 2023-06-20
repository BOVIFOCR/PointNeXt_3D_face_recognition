from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path


# BERNARDO
class TreeLFW:
    
    def load_img_pairs_from_protocol_file(self, dataset_path='', protocol_file_path='pairs.txt', file_ext='.jpg'):
        pos_pair_label = '1'
        neg_pair_label = '0'
        all_pairs_paths_label = []
        folds_indexes = []

        with open(protocol_file_path, 'r') as fp:
            all_lines = [line.rstrip('\n') for line in fp.readlines()]
            # print('all_lines:', all_lines)
            if len(all_lines[0].split('\t')) > 1:
                num_folds, fold_size = int(all_lines[0].split('\t')[0]), int(all_lines[0].split('\t')[1])
            else:
                num_folds = 1
                fold_size = int(all_lines[0].split('\t')[0])

            total_num_pairs = num_folds*(fold_size*2)

            global_pair_idx = 1
            while global_pair_idx < total_num_pairs:
                start_fold_idx = global_pair_idx-1
                end_fold_idx = start_fold_idx + (fold_size*2)

                pos_pairs_paths = []
                for _ in range(1, fold_size+1):
                    pos_pair = all_lines[global_pair_idx].split('\t')   # Abel_Pacheco	1	4
                    subj_name, index1, index2 = pos_pair
                    assert index1 != index2

                    # path_sample1 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index1.zfill(4), '*'+file_ext))[0]
                    # path_sample2 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index2.zfill(4), '*'+file_ext))[0]
                    path_sample1 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index1.zfill(4)+file_ext))[0]
                    path_sample2 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index2.zfill(4)+file_ext))[0]

                    pos_pair = (pos_pair_label, path_sample1, path_sample2)
                    pos_pairs_paths.append(pos_pair)
                    global_pair_idx += 1
                all_pairs_paths_label += pos_pairs_paths

                neg_pairs_paths = []
                for _ in range(1, fold_size+1):
                    neg_pair = all_lines[global_pair_idx].split('\t')   # Abdel_Madi_Shabneh	1	Dean_Barker	1
                    subj_name1, index1, subj_name2, index2 = neg_pair
                    assert subj_name1 != subj_name2

                    # path_sample1 = glob(os.path.join(dataset_path, subj_name1, subj_name1+'_'+index1.zfill(4), '*'+file_ext))[0]
                    # path_sample2 = glob(os.path.join(dataset_path, subj_name2, subj_name2+'_'+index2.zfill(4), '*'+file_ext))[0]
                    path_sample1 = glob(os.path.join(dataset_path, subj_name1, subj_name1+'_'+index1.zfill(4)+file_ext))[0]
                    path_sample2 = glob(os.path.join(dataset_path, subj_name2, subj_name2+'_'+index2.zfill(4)+file_ext))[0]

                    neg_pair = (neg_pair_label, path_sample1, path_sample2)
                    neg_pairs_paths.append(neg_pair)
                    global_pair_idx += 1
                all_pairs_paths_label += neg_pairs_paths

                folds_indexes.append((start_fold_idx, end_fold_idx))

            return all_pairs_paths_label, folds_indexes, pos_pair_label, neg_pair_label
