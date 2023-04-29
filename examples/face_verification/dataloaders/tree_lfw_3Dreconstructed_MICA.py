from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path


# BERNARDO
class TreeLFW_3DReconstructedMICA:
    
    def get_all_sub_folders(self, dir_path=''):
        return sorted(glob(dir_path + '/*/*/'))

    def get_sub_folders_one_level(self, dir_path=''):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)

    def get_all_pointclouds_paths(self, dir_path, pc_ext='.ply'):
        all_sub_folders = self.get_all_sub_folders(dir_path)
        all_pc_paths = []
        all_pc_subjects = []
        for sub_folder_pointcloud in all_sub_folders:
            pc_paths = sorted(glob(sub_folder_pointcloud + '/*' + pc_ext))
            # print('pc_paths:', pc_paths)
            pc_subjects = [pc_path.split('/')[-3] for pc_path in pc_paths]
            all_pc_paths += pc_paths
            all_pc_subjects += pc_subjects
        return all_pc_paths, all_pc_subjects

    def count_samples_per_subject(self, pc_paths_list=[''], pc_ext='.ply'):
        unique_subjects_names = []
        samples_per_subject = []
        for pc_path in sorted(pc_paths_list):
            data_path = pc_path.split('/')
            if pc_ext in data_path[-1]:
                subject_name = data_path[-3]
                if not subject_name in unique_subjects_names:
                    samples_per_subject.append(0)
                    unique_subjects_names.append(subject_name)
                samples_per_subject[-1] += 1
        assert len(unique_subjects_names) == len(samples_per_subject)
        return unique_subjects_names, samples_per_subject

    def get_all_pointclouds_paths_count(self, dir_path, pc_ext='.ply'):
        all_pc_paths, all_pc_subjects = self.get_all_pointclouds_paths(dir_path, pc_ext)
        unique_subjects_names, samples_per_subject = self.count_samples_per_subject(all_pc_paths, pc_ext)
        return all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject

    def filter_paths_by_minimum_samples(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext='.ply', min_samples=2, max_samples=-1):
        filtered_pc_paths = []
        filtered_pc_subjects = []
        filtered_subjects_names = []
        filtered_samples_per_subject = []
        selected_samples_per_subject = [0] * len(unique_subjects_names)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            if samples_per_subject[unique_subjects_names.index(pc_subj)] >= min_samples and \
               (max_samples==-1 or selected_samples_per_subject[unique_subjects_names.index(pc_subj)] < max_samples):
                filtered_pc_paths.append(pc_path)
                filtered_pc_subjects.append(pc_subj)
                if not pc_subj in filtered_subjects_names:   # run once per subject
                    filtered_subjects_names.append(pc_subj)
                selected_samples_per_subject[unique_subjects_names.index(pc_subj)] += 1
        # filtered_samples_per_subject.append(samples_per_subject[unique_subjects_names.index(pc_subj)])
        filtered_samples_per_subject = [selected_samples_per_subject[unique_subjects_names.index(pc_subj)] for pc_subj in filtered_subjects_names]
        # print('selected_samples_per_subject:', selected_samples_per_subject)      
        return filtered_pc_paths, filtered_pc_subjects, filtered_subjects_names, filtered_samples_per_subject

    def load_filter_organize_pointclouds_paths(self, dir_path, pc_ext='.ply', min_samples=2, max_samples=-1):
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.get_all_pointclouds_paths_count(dir_path, pc_ext)
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext, min_samples, max_samples)
        subjects_with_pc_paths = [()] * len(all_pc_paths)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            subjects_with_pc_paths[i] = (pc_subj, pc_path)
        # print('samples_per_subject:', samples_per_subject)
        return subjects_with_pc_paths, unique_subjects_names, samples_per_subject

    def load_pairs_samples_protocol_from_file(self, protocol_file_path='pairsDevTrain.txt', dataset_path='', file_ext='.ply'):
        pos_pair_label = 1
        neg_pair_label = 0
        all_pos_pairs_paths = []
        all_neg_pairs_paths = []

        with open(protocol_file_path, 'r') as fp:
            all_lines = [line.rstrip('\n') for line in fp.readlines()]
            # print('all_lines:', all_lines)
            num_pos_pairs = int(all_lines[0])
            for i in range(1, num_pos_pairs+1):
                pos_pair = all_lines[i].split('\t')   # Aaron_Peirsol	1	2
                # print('pos_pair:', pos_pair)
                subj_name, index1, index2 = pos_pair
                path_sample1 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index1.zfill(4), '*'+file_ext))[0]
                path_sample2 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index2.zfill(4), '*'+file_ext))[0]
                # pos_pair = (subj_name, pos_pair_label, path_sample1, path_sample2)
                pos_pair = (pos_pair_label, path_sample1, path_sample2)
                all_pos_pairs_paths.append(pos_pair)
                # print('path_sample1:', path_sample1)
                # print('path_sample2:', path_sample2)
                # print('pos_pair:', pos_pair)

            for i in range(num_pos_pairs+1, len(all_lines)):
                neg_pair = all_lines[i].split('\t')   # AJ_Cook	1	Marsha_Thomason	1
                # print('neg_pair:', neg_pair)
                subj_name1, index1, subj_name2, index2 = neg_pair
                path_sample1 = glob(os.path.join(dataset_path, subj_name1, subj_name1+'_'+index1.zfill(4), '*'+file_ext))[0]
                path_sample2 = glob(os.path.join(dataset_path, subj_name2, subj_name2+'_'+index2.zfill(4), '*'+file_ext))[0]
                neg_pair = (neg_pair_label, path_sample1, path_sample2)
                all_neg_pairs_paths.append(neg_pair)
                # sys.exit(0)
            return all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label


    def load_all_pairs_samples_from_protocol_file(self, dataset_path='', protocol_file_path='pairs.txt', file_ext='.npy'):
        pos_pair_label = '1'
        neg_pair_label = '0'
        all_pairs_paths_label = []

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

                pos_pairs_paths = []
                for _ in range(1, fold_size+1):
                    pos_pair = all_lines[global_pair_idx].split('\t')   # Aaron_Peirsol	1	2
                    subj_name, index1, index2 = pos_pair
                    assert index1 != index2
                    path_sample1 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index1.zfill(4), '*'+file_ext))[0]
                    path_sample2 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index2.zfill(4), '*'+file_ext))[0]
                    pos_pair = (pos_pair_label, path_sample1, path_sample2)
                    pos_pairs_paths.append(pos_pair)
                    global_pair_idx += 1
                all_pairs_paths_label += pos_pairs_paths

                neg_pairs_paths = []
                for _ in range(1, fold_size+1):
                    neg_pair = all_lines[global_pair_idx].split('\t')   # AJ_Cook	1	Marsha_Thomason	1
                    subj_name1, index1, subj_name2, index2 = neg_pair
                    assert subj_name1 != subj_name2
                    path_sample1 = glob(os.path.join(dataset_path, subj_name1, subj_name1+'_'+index1.zfill(4), '*'+file_ext))[0]
                    path_sample2 = glob(os.path.join(dataset_path, subj_name2, subj_name2+'_'+index2.zfill(4), '*'+file_ext))[0]
                    neg_pair = (neg_pair_label, path_sample1, path_sample2)
                    neg_pairs_paths.append(neg_pair)
                    global_pair_idx += 1
                all_pairs_paths_label += neg_pairs_paths

            return all_pairs_paths_label, pos_pair_label, neg_pair_label


if __name__ == '__main__':
    dataset_path = '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw'
    
    file_ext='.ply'
    
    min_samples=1
    # min_samples=3

    max_samples=-1
    # max_samples=100

    log_scale = True
    # log_scale = False

    # all_pc_paths, all_pc_subjects = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths(dir_path=dataset_path, pc_ext=file_ext)
    # unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().count_samples_per_subject(all_pc_paths, pc_ext=file_ext)
    # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # for i, unique_subj_name in enumerate(unique_subjects_names):
    #     print('unique_subj_name', unique_subj_name, '    samples_per_subject:', samples_per_subject[i])
    # sys.exit(0)
    
    # print('Searching all files ending with \'' + file_ext + '\'...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths_count(dataset_path, file_ext)
    # # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    # #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # # sys.exit(0)
    # # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    # #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # # sys.exit(0)

    # print('Searching all files ending with \'' + file_ext + '\'...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, file_ext, min_samples)
    # # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    # #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # # sys.exit(0)
    # # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    # #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # # print('len(unique_subj_name):', len(unique_subjects_names), '    len(samples_per_subj):', len(samples_per_subject))
    # # sys.exit(0)
    
    # print('Searching all files ending with \'' + file_ext + '\'...')
    # subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().load_filter_organize_pointclouds_paths(dataset_path, file_ext, min_samples, max_samples)
    # # for subj_pc_path in subjects_with_pc_paths:
    # #     print('subj_pc_path:', subj_pc_path)
    # # # sys.exit(0)

    # # unique_subjects_names = ['AAAAA', 'BBBB', 'CCCC', 'DDDD']
    # # samples_per_subject = [5, 2, 2, 1]
    # title = 'Dataset LFW - Samples per Subject'
    # subtitle = '(min_samples='+str(min_samples)+', max_samples='+str(max_samples)+')'
    # path_image = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/samples_per_subject_lfw_dataset_minsamples='+str(min_samples)+'_maxsamples='+str(max_samples)+'.png'
    # plots_fr_pointnet2.plot_samples_per_class_and_histogram(unique_subjects_names, samples_per_subject, log_scale, title=title, subtitle=subtitle, path_image=path_image, show_fig=False, save_fig=True)
    
    protocol_file_path = '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/pairsDevTrain.txt'
    all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label = TreeLFW_3DReconstructedMICA().load_pairs_samples_protocol_from_file(protocol_file_path, dataset_path, file_ext)
    for pos_pairs in all_pos_pairs_paths:
        print('pos_pairs:', pos_pairs)
    sys.exit(0)
    # for neg_pairs in all_neg_pairs_paths:
    #     print('neg_pairs:', neg_pairs)
    # sys.exit(0)