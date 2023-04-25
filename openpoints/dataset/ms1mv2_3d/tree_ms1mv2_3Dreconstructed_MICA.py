from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path
import random


# BERNARDO
class TreeMS1MV2_3DReconstructedMICA:
    
    def get_all_sub_folders(self, dir_path='', dir_level=2):
        return sorted(glob(dir_path + '/*'*dir_level))

    def get_sub_folders_one_level(self, dir_path=''):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)
    
    
    def get_all_pointclouds_paths(self, dir_path, dir_level=2, pc_ext='.ply'):
        all_sub_folders = self.get_all_sub_folders(dir_path, dir_level)
        all_pc_paths = []
        all_pc_subjects = []
        # print('all_sub_folders:', all_sub_folders)
        # print('len(all_sub_folders):', len(all_sub_folders))
        for sub_folder_pointcloud in all_sub_folders:
            pc_paths = sorted(glob(sub_folder_pointcloud + '/*' + pc_ext))
            # print('pc_paths:', pc_paths)
            assert len(pc_paths) > 0
            pc_subjects = [pc_path.split('/')[-3] for pc_path in pc_paths]
            # print('pc_subjects:', pc_subjects)
            assert len(pc_subjects) > 0
            # print('----------------------')
            # raw_input('PAUSED')
            all_pc_paths += pc_paths
            all_pc_subjects += pc_subjects
        
        assert len(all_pc_paths) > 0
        assert len(all_pc_subjects) > 0
        return all_pc_paths, all_pc_subjects

    '''
    # TEMP WHILE FACES ARE BEING RECONSTRUCTED BY MICA
    def get_all_pointclouds_paths(self, dir_path, dir_level=2, pc_ext='.ply'):
        all_sub_folders = self.get_all_sub_folders(dir_path, dir_level)
        all_pc_paths = []
        all_pc_subjects = []
        # print('all_sub_folders:', all_sub_folders)
        # print('len(all_sub_folders):', len(all_sub_folders))
        for sub_folder_pointcloud in all_sub_folders:
            # print('sub_folder_pointcloud:', sub_folder_pointcloud)
            pc_paths = sorted(glob(sub_folder_pointcloud + '/*' + pc_ext))
            # print('pc_paths:', pc_paths)
            if len(pc_paths) > 0:    # Added to prevent error
                # print('pc_paths:', pc_paths)
                assert len(pc_paths) > 0
                pc_subjects = [pc_path.split('/')[-(dir_level+1)] for pc_path in pc_paths]
                # print('pc_subjects:', pc_subjects)
                assert len(pc_subjects) > 0
                # print('----------------------')
                # raw_input('PAUSED')
                all_pc_paths += pc_paths
                all_pc_subjects += pc_subjects
        
        assert len(all_pc_paths) > 0
        assert len(all_pc_subjects) > 0
        return all_pc_paths, all_pc_subjects
    '''

    def count_samples_per_subject(self, pc_paths_list=[''], dir_level=2, pc_ext='.ply'):
        unique_subjects_names = []
        samples_per_subject = []
        indexes_samples = []
        for i, pc_path in enumerate(sorted(pc_paths_list)):
            # print('%d/%d - %s' % (i, len(pc_paths_list), pc_path), end='\r')
            print('%d/%d - \'%s\'' % (i, len(pc_paths_list), pc_path), end='\r')
            data_path = pc_path.split('/')
            if pc_ext in data_path[-1]:
                subject_name = data_path[-(dir_level+1)]
                if not subject_name in unique_subjects_names:
                    samples_per_subject.append(0)
                    unique_subjects_names.append(subject_name)
                    indexes_samples.append([i, i-1])    # i=begin, i-1=end

                samples_per_subject[-1] += 1
                indexes_samples[-1][1] += 1   # increment end index
        assert len(unique_subjects_names) == len(samples_per_subject)
        print()
        return unique_subjects_names, samples_per_subject, indexes_samples

    def get_all_pointclouds_paths_count(self, dir_path, dir_level=2, pc_ext='.ply'):
        print('get_all_pointclouds_paths_count() - getting all pointclouds paths ...')
        all_pc_paths, all_pc_subjects = self.get_all_pointclouds_paths(dir_path, dir_level, pc_ext)
        print('get_all_pointclouds_paths_count() - counting samples per subject ...')
        unique_subjects_names, samples_per_subject, indexes_samples = self.count_samples_per_subject(all_pc_paths, dir_level, pc_ext)
        return all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples

    def filter_paths_by_minimum_samples(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext='.ply', min_samples=2, max_samples=-1):
        filtered_pc_paths = []
        filtered_pc_subjects = []
        filtered_subjects_names = []
        filtered_samples_per_subject = []
        selected_samples_per_subject = [0] * len(unique_subjects_names)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            # print('%d/%d - %s' % (i+1, len(all_pc_paths), pc_path), end='\r')
            print('%d/%d - \'%s\'' % (i+1, len(all_pc_paths), pc_path), end='\r')
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
        print()
        return filtered_pc_paths, filtered_pc_subjects, filtered_subjects_names, filtered_samples_per_subject

    def load_filter_organize_pointclouds_paths(self, dir_path, dir_level=2, pc_ext='.ply', min_samples=2, max_samples=-1):
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples = self.get_all_pointclouds_paths_count(dir_path, dir_level=dir_level, pc_ext=pc_ext)
        print('load_filter_organize_pointclouds_paths() - filtering paths by min and max samples ...')
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext, min_samples, max_samples)
        subjects_with_pc_paths = [()] * len(all_pc_paths)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            subjects_with_pc_paths[i] = (pc_subj, pc_path)
        # print('samples_per_subject:', samples_per_subject)
        return subjects_with_pc_paths, unique_subjects_names, samples_per_subject

    def load_pairs_samples_protocol_from_file(self, protocol_file_path='pairsDevTrain.txt', dataset_path='', file_ext='.ply'):
        pos_pair_label = '1'
        neg_pair_label = '0'
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


    def make_pairs_global_indexes(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples=True):
        random.seed(440)
        
        def choose_random_sample(begin, end, amount=1):
            return random.sample(range(begin, end+1), amount)[0]

        def make_random_pair(begin, end, amount=2):
            return random.sample(range(begin, end+1), amount)

        def is_pair_valid(avail_all_pc_paths, idx1, idx2):
            if avail_all_pc_paths[idx1] == True and avail_all_pc_paths[idx2] == True:
                return True
            return False
        
        def is_pair_duplicate(pair_search, all_pairs):
            assert len(pair_search) == 2
            found_indexes = [idx for idx, pair in enumerate(all_pairs) if pair_search[0] in pair and pair_search[1] in pair]
            if len(found_indexes) > 0:
                return True
            return False
        
        pos_pairs = [None] * num_pos_pairs
        neg_pairs = [None] * num_neg_pairs
        avail_all_pc_paths = [True] * len(all_pc_paths)

        # Make positive pairs
        rand_subj_idx = random.sample(range(0, len(unique_subjects_names)), len(unique_subjects_names))
        pair_idx = 0
        subj_idx = 0
        while pair_idx < num_pos_pairs:
            if samples_per_subject[rand_subj_idx[subj_idx]] > 1:   # for positive pairs, use only subjects containing 2 or more samples
                begin_subj, end_subj = indexes_samples[rand_subj_idx[subj_idx]]

                one_pos_pair_idx = make_random_pair(begin_subj, end_subj, amount=2)
                while is_pair_duplicate(one_pos_pair_idx, pos_pairs[:pair_idx]):
                    print('Duplicate positive pair found:', one_pos_pair_idx, '    formed pairs:', str(pair_idx)+'/'+str(num_pos_pairs))
                    subj_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]
                    if samples_per_subject[rand_subj_idx[subj_idx]] > 1:
                        begin_subj, end_subj = indexes_samples[rand_subj_idx[subj_idx]]
                        one_pos_pair_idx = make_random_pair(begin_subj, end_subj, amount=2)

                if not reuse_samples:
                    while not is_pair_valid(avail_all_pc_paths, one_pos_pair_idx[0], one_pos_pair_idx[1]):
                        one_pos_pair_idx = make_random_pair(begin_subj, end_subj, amount=2)
                
                avail_all_pc_paths[one_pos_pair_idx[0]], avail_all_pc_paths[one_pos_pair_idx[1]] = False, False
                pos_pairs[pair_idx] = [unique_subjects_names[rand_subj_idx[subj_idx]], one_pos_pair_idx[0], one_pos_pair_idx[1]]
                # print(str(pair_idx)+'/'+str(num_pos_pairs) + '    subject_name:', unique_subjects_names[rand_subj_idx[subj_idx]], '    samples_per_subject:', samples_per_subject[rand_subj_idx[subj_idx]], '    indexes_samples:', indexes_samples[rand_subj_idx[subj_idx]], '    pos_pairs[pair_idx]:', pos_pairs[pair_idx])
                # raw_input('PAUSED')
                pair_idx += 1
            subj_idx += 1
            if subj_idx >= len(unique_subjects_names):
                subj_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]

        # Make negative pairs
        rand_subj_idx = random.sample(range(0, len(unique_subjects_names)), len(unique_subjects_names))
        pair_idx = 0
        subj1_idx, subj2_idx = 0, 1
        while pair_idx < num_neg_pairs:
            begin_subj1, end_subj1 = indexes_samples[rand_subj_idx[subj1_idx]]
            begin_subj2, end_subj2 = indexes_samples[rand_subj_idx[subj2_idx]]
            
            one_neg_pair_idx = [choose_random_sample(begin_subj1, end_subj1, amount=1), choose_random_sample(begin_subj2, end_subj2, amount=1)]
            while is_pair_duplicate(one_neg_pair_idx, neg_pairs[:pair_idx]):
                print('Duplicate negative pair found:', one_neg_pair_idx, '    formed pairs:', str(pair_idx)+'/'+str(num_neg_pairs))
                subj1_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]
                subj2_idx = subj1_idx+1
                begin_subj1, end_subj1 = indexes_samples[rand_subj_idx[subj1_idx]]
                begin_subj2, end_subj2 = indexes_samples[rand_subj_idx[subj2_idx]]
                one_neg_pair_idx = [choose_random_sample(begin_subj1, end_subj1, amount=1), choose_random_sample(begin_subj2, end_subj2, amount=1)]
            
            if not reuse_samples:
                while not is_pair_valid(avail_all_pc_paths, one_neg_pair_idx[0], one_neg_pair_idx[1]):
                    one_neg_pair_idx = [choose_random_sample(begin_subj1, end_subj1, amount=1), choose_random_sample(begin_subj2, end_subj2, amount=1)]

            avail_all_pc_paths[one_neg_pair_idx[0]], avail_all_pc_paths[one_neg_pair_idx[1]] = False, False
            neg_pairs[pair_idx] = [unique_subjects_names[rand_subj_idx[subj1_idx]], one_neg_pair_idx[0], unique_subjects_names[rand_subj_idx[subj2_idx]], one_neg_pair_idx[1]]
            # print('subject_name1:', unique_subjects_names[rand_subj_idx[subj1_idx]], '    samples_per_subject1:', samples_per_subject[rand_subj_idx[subj1_idx]], '    indexes_samples1:', indexes_samples[rand_subj_idx[subj1_idx]])
            # print('subject_name2:', unique_subjects_names[rand_subj_idx[subj2_idx]], '    samples_per_subject2:', samples_per_subject[rand_subj_idx[subj2_idx]], '    indexes_samples2:', indexes_samples[rand_subj_idx[subj2_idx]])
            # print(str(pair_idx)+'/'+str(num_neg_pairs) + '    neg_pairs[pair_idx]:', neg_pairs[pair_idx])            
            # raw_input('PAUSED')
            # print('--------------------------')
            pair_idx += 1
            subj1_idx += 2
            subj2_idx += 2

            if subj2_idx >= len(unique_subjects_names):
                subj1_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]
                subj2_idx = subj1_idx+1

        return pos_pairs, neg_pairs


    def make_pairs_indexes_lfw_format(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples=True):
        pos_pairs, neg_pairs = self.make_pairs_global_indexes(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples)

        for i in range(len(pos_pairs)):    # pos_pairs[i] = [0=subj_name, 1=global_idx1, 2=global_idx2]
            file_name_sample1 = all_pc_paths[pos_pairs[i][1]].split('/')[-2]
            file_name_sample2 = all_pc_paths[pos_pairs[i][2]].split('/')[-2]
            pos_pairs[i][1] = file_name_sample1
            pos_pairs[i][2] = file_name_sample2
            # print('pos_pairs[i]:', pos_pairs[i])            
            # raw_input('PAUSED')
            # print('--------------------------')
        
        for i in range(len(neg_pairs)):    # neg_pairs[i] = [0=subj_name1, 1=global_idx1, 2=subj_name2, 3=global_idx2]
            file_name_sample1 = all_pc_paths[neg_pairs[i][1]].split('/')[-2]
            file_name_sample2 = all_pc_paths[neg_pairs[i][3]].split('/')[-2]
            neg_pairs[i][1] = file_name_sample1
            neg_pairs[i][3] = file_name_sample2
            # print('neg_pairs[i]:', neg_pairs[i])            
            # raw_input('PAUSED')
            # print('--------------------------')
        
        return pos_pairs, neg_pairs


    def make_pairs_labels_with_paths(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples=True):
        pos_pairs, neg_pairs = self.make_pairs_global_indexes(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples)
        pos_pair_label = '1'
        neg_pair_label = '0'

        all_pos_pairs_paths = []
        all_neg_pairs_paths = []

        for i, pos_pair in enumerate(pos_pairs):
            # print('pair '+str(i)+'/'+str(len(pos_pairs)), '   pos_pair:', pos_pair)
            subj_name, index1, index2 = pos_pair
            path_sample1 = all_pc_paths[index1]
            path_sample2 = all_pc_paths[index2]
            all_pos_pairs_paths.append((pos_pair_label, path_sample1, path_sample2))
            # print('all_pos_pairs_paths[-1]:', all_pos_pairs_paths[-1])            
            # raw_input('PAUSED')
            # print('--------------------------')

        for i, neg_pair in enumerate(neg_pairs):
            # print('pair '+str(i)+'/'+str(len(pos_pairs)), '   neg_pair:', neg_pair)
            subj_name1, index1, subj_name2, index2 = neg_pair
            path_sample1 = all_pc_paths[index1]
            path_sample2 = all_pc_paths[index2]
            all_neg_pairs_paths.append((neg_pair_label, path_sample1, path_sample2))
            # print('all_neg_pairs_paths[-1]:', all_neg_pairs_paths[-1])            
            # raw_input('PAUSED')
            # print('--------------------------')
        
        # return all_pos_pairs_paths, all_neg_pairs_paths
        return all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label


    def save_pairs_txt_file_lfw_format(self, pos_pairs_format_lfw, neg_pairs_format_lfw, perc_train, perc_test, pairs_file_path):
        with open(pairs_file_path, 'w') as file:
            num_pos_pairs_train = int(round(len(pos_pairs_format_lfw) * perc_train))
            num_neg_pairs_train = int(round(len(neg_pairs_format_lfw) * perc_train))
            file.write(str(num_pos_pairs_train) + '\ttrain positive pairs' + '\n')
            for i, pos_pair in enumerate(pos_pairs_format_lfw[:num_pos_pairs_train]):
                file.write(str(pos_pair[0]) + '\t' + str(pos_pair[1]) + '\t' + str(pos_pair[2]) + '\n')
            
            file.write(str(num_neg_pairs_train) + '\ttrain negative pairs' + '\n')
            for i, neg_pair in enumerate(neg_pairs_format_lfw[:num_neg_pairs_train]):
                file.write(str(neg_pair[0]) + '\t' + str(neg_pair[1]) + '\t' + str(neg_pair[2]) + '\t' + str(neg_pair[3]) + '\n')


            num_pos_pairs_test = int(round(len(pos_pairs_format_lfw) * perc_test))
            num_neg_pairs_test = int(round(len(neg_pairs_format_lfw) * perc_test))
            file.write(str(num_pos_pairs_test) + '\ttest positive pairs' + '\n')
            for i, pos_pair in enumerate(pos_pairs_format_lfw[num_pos_pairs_test:]):
                file.write(str(pos_pair[0]) + '\t' + str(pos_pair[1]) + '\t' + str(pos_pair[2]) + '\n')
            
            file.write(str(num_neg_pairs_test) + '\ttest negative pairs' + '\n')
            for i, neg_pair in enumerate(neg_pairs_format_lfw[num_neg_pairs_test:]):
                file.write(str(neg_pair[0]) + '\t' + str(neg_pair[1]) + '\t' + str(neg_pair[2]) + '\t' + str(neg_pair[3]) + '\n')


    def check_duplicate_pairs(self, pos_pairs, neg_pairs):
        num_repeated_pos_pairs = 0
        num_repeated_neg_pairs = 0
        for pair_search in pos_pairs:
            found_indexes = [idx for idx, pair in enumerate(pos_pairs) if pair[1] in pair_search and pair[2] in pair_search]
            num_repeated_pos_pairs += len(found_indexes) - 1  # consider only repetitions
            if len(found_indexes) > 1:
                duplicate_pairs = [pos_pairs[fi] for fi in found_indexes]
                print('pair_search:', pair_search)
                print('duplicate_pairs:', duplicate_pairs)
                print('found_indexes:', found_indexes)
                print('----------------')
        
        for pair_search in neg_pairs:
            found_indexes = [idx for idx, pair in enumerate(neg_pairs) if pair[1] in pair_search and pair[3] in pair_search]
            num_repeated_pos_pairs += len(found_indexes) - 1  # consider only repetitions
            if len(found_indexes) > 1:
                duplicate_pairs = [neg_pairs[fi] for fi in found_indexes]
                print('pair_search:', pair_search)
                print('duplicate_pairs:', duplicate_pairs)
                print('found_indexes:', found_indexes)
                print('----------------')
        
        print('pos_pairs:', len(pos_pairs), '    num_repeated_pos_pairs:', num_repeated_pos_pairs)
        print('neg_pairs:', len(neg_pairs), '    num_repeated_neg_pairs:', num_repeated_neg_pairs)


if __name__ == '__main__':
    dataset_path = '/home/bjgbiesseck/GitHub/MICA/demo/output/MS-Celeb-1M/ms1m-retinaface-t1/images'

    dir_level = 2
    
    # file_ext = '.ply'
    # file_ext = '_centralized_nosetip.ply'
    file_ext = '_centralized-nosetip_with-normals_filter-radius=100.npy'
    
    min_samples = 1
    # min_samples = 3

    max_samples = -1
    # max_samples = 100

    log_scale = True
    # log_scale = False

    # num_pos_pairs, num_neg_pairs = 10, 10
    num_pos_pairs, num_neg_pairs = 50, 50
    # num_pos_pairs, num_neg_pairs = 2000, 2000
    # num_pos_pairs, num_neg_pairs = 10000, 10000
    # num_pos_pairs, num_neg_pairs = 25000, 25000
    # num_pos_pairs, num_neg_pairs = 50000, 50000
    # num_pos_pairs, num_neg_pairs = 100000, 100000
    # num_pos_pairs, num_neg_pairs = 200000, 200000

    reuse_samples = True
    # reuse_samples = False

    # print('Searching all files ending with \'' + file_ext + '\' in \'' + dataset_path + '\' ...')
    # all_pc_paths, all_pc_subjects = TreeMS1MV2_3DReconstructedMICA().get_all_pointclouds_paths(dir_path=dataset_path, dir_level=dir_level, pc_ext=file_ext)
    # unique_subjects_names, samples_per_subject = TreeMS1MV2_3DReconstructedMICA().count_samples_per_subject(all_pc_paths, pc_ext=file_ext)
    # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # print('len(all_pc_paths):', len(all_pc_paths), '    len(all_pc_subjects):', len(all_pc_subjects))
    # raw_input('Paused, press ENTER')
    # for unique_subj_name, samp_per_subj in zip(unique_subjects_names, samples_per_subject):
    #     print('unique_subj_name', unique_subj_name, '    samp_per_subj:', samp_per_subj)
    # print('len(unique_subjects_names):', len(unique_subjects_names), '    len(samples_per_subject):', len(samples_per_subject))
    # sys.exit(0)
    
    print('Searching all files ending with \'' + file_ext + '\' in \'' + dataset_path + '\' ...')
    all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples = TreeMS1MV2_3DReconstructedMICA().get_all_pointclouds_paths_count(dataset_path, dir_level, file_ext)
    # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # print('len(all_pc_paths):', len(all_pc_paths), '    len(all_pc_subjects):', len(all_pc_subjects))
    # raw_input('Paused, press ENTER')
    # for unique_subj_name, samp_per_subj, indexes_samp in zip(unique_subjects_names, samples_per_subject, indexes_samples):
    #     print('unique_subj_name', unique_subj_name, '    samp_per_subj:', samp_per_subj, '    indexes_samp:', indexes_samp)
    #     # raw_input('Paused, press ENTER')
    # print('len(unique_subjects_names):', len(unique_subjects_names), '    len(samples_per_subject):', len(samples_per_subject), '    len(indexes_samples):', len(indexes_samples))
    # sys.exit(0)

    # print('Making train and test pairs...')
    # pos_pairs, neg_pairs = TreeMS1MV2_3DReconstructedMICA().make_pairs_global_indexes(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples)
    # pos_pairs_format_labels_paths, neg_pairs_labels_paths, pos_pair_label, neg_pair_label = TreeMS1MV2_3DReconstructedMICA().make_pairs_labels_with_paths(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples)
    
    print('Making train and test pairs to save in file...')
    pos_pairs_format_lfw, neg_pairs_format_lfw = TreeMS1MV2_3DReconstructedMICA().make_pairs_indexes_lfw_format(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples)
    perc_train = 0.8
    perc_test = 1.0 - perc_train
    total_train_pairs = int(round((num_pos_pairs + num_neg_pairs) * perc_train))
    total_test_pairs = int(round((num_pos_pairs + num_neg_pairs) * perc_test))
    pairs_file_path = 'ms1mv2_retinaface_subj_10408_pairs_' + str(total_train_pairs+total_test_pairs) + '_train_' + str(total_train_pairs) + '_test_' + str(total_test_pairs) + '.txt'
    print('Saving train and test pairs to file \'' + pairs_file_path + '\'' + '...')
    TreeMS1MV2_3DReconstructedMICA().save_pairs_txt_file_lfw_format(pos_pairs_format_lfw, neg_pairs_format_lfw, perc_train, perc_test, pairs_file_path)
    print('Saved!')





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
    
    # protocol_file_path = '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/pairsDevTrain.txt'
    # all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label = TreeLFW_3DReconstructedMICA().load_pairs_samples_protocol_from_file(protocol_file_path, dataset_path, file_ext)
    # for pos_pairs in all_pos_pairs_paths:
    #     print('pos_pairs:', pos_pairs)
    # sys.exit(0)
    # # for neg_pairs in all_neg_pairs_paths:
    # #     print('neg_pairs:', neg_pairs)
    # # sys.exit(0)