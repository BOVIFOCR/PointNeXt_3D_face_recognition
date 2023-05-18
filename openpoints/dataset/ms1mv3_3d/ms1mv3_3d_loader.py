"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import sys, os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from math import floor
import pickle

from .tree_ms1mv3_3Dreconstructed_MICA import TreeMS1MV3_3DReconstructedMICA


@DATASETS.register_module()
class MS1MV3_3D(Dataset):
    
    def __init__(self,
                 n_classes,
                 num_points,
                 data_dir="",
                 split='train',
                 transform=None
                 ):

        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        self.num_points = num_points
        self.transform = transform

        self.DATA_PATH = data_dir
        self.n_classes = n_classes
        
        dir_level=2
        min_samples=2
        max_samples=-1
        file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'
        # file_ext = '.ply'

        logging.info(f'loading dataset...')

        paths_file_name = 'paths_' + self.DATA_PATH.split('/')[-1] + '_min-samples=' + str(min_samples) + '_max-samples=' + str(max_samples) + 'file-ext=\'' + file_ext + '\'.pkl'
        path_paths_file = self.DATA_PATH + '/' + paths_file_name

        if not os.path.isfile(path_paths_file):
            subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeMS1MV3_3DReconstructedMICA().load_filter_organize_pointclouds_paths(self.DATA_PATH, dir_level, file_ext, min_samples, max_samples)

            paths_dict = {'subjects_with_pc_paths': subjects_with_pc_paths, 'unique_subjects_names': unique_subjects_names, 'samples_per_subject': samples_per_subject}
            with open(path_paths_file, 'wb') as file:
                print('MS1MV3_3D.__init__ - Saving point cloud paths:', path_paths_file)
                pickle.dump(paths_dict, file)

        else:
            with open(path_paths_file, 'rb') as file:
                print('MS1MV3_3D.__init__ - Reading point cloud paths:', path_paths_file)
                paths_dict = pickle.load(file)
                subjects_with_pc_paths = paths_dict['subjects_with_pc_paths']
                unique_subjects_names = paths_dict['unique_subjects_names']
                samples_per_subject = paths_dict['samples_per_subject']

        assert len(unique_subjects_names) == len(samples_per_subject)

        self.cat = unique_subjects_names    # Bernardo
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        logging.info(f'==> sucessfully loaded {self.partition} data')

        self.data = []

        if split=='train':
            last_index = 0
            for samp_per_subj, uniq_subj_name in zip(samples_per_subject, unique_subjects_names):
                amount_train_samples_subj = int(floor(samp_per_subj * 0.8))
                train_subj_with_paths = []
                for i in range(last_index, len(subjects_with_pc_paths)):
                    if subjects_with_pc_paths[i][0] == uniq_subj_name:
                        train_subj_with_paths.append(subjects_with_pc_paths[i])
                        if len(train_subj_with_paths) == amount_train_samples_subj:
                            last_index = i
                            break
                assert len(train_subj_with_paths) == amount_train_samples_subj
                self.data += train_subj_with_paths

        elif split=='test':
            last_index = 0
            for samp_per_subj, uniq_subj_name in zip(samples_per_subject, unique_subjects_names):
                amount_train_samples_subj = int(floor(samp_per_subj * 0.8))
                amount_test_samples_subj = samp_per_subj - amount_train_samples_subj
                test_subj_with_paths = []
                for i in range(last_index+amount_train_samples_subj, len(subjects_with_pc_paths)):
                    if subjects_with_pc_paths[i][0] == uniq_subj_name:
                        test_subj_with_paths.append(subjects_with_pc_paths[i])
                        if len(test_subj_with_paths) == amount_test_samples_subj:
                            last_index = i+1
                            break
                assert len(test_subj_with_paths) == amount_test_samples_subj
                self.data += test_subj_with_paths


    def __getitem__(self, item):
        pointcloud, cls = self._get_item(item)
        pointcloud = pointcloud[:self.num_points]
        label = cls

        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }

        if self.transform is not None:
            data = data
            #print("transforming some data")
            #data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']

        return data

    def __len__(self):
        return len(self.data)

    @property
    def num_classes(self):
        return np.max(self.label) + 1



    def _get_item(self, index): 
        fn = self.data[index]
        cls = self.classes[self.data[index][0]]
        #cls = np.array([cls]).astype(np.int32)
        cls = np.array([cls])
        # Bernardo
        #print('synthetic_faces_gpmm_dataset.py: _get_item(): loading file:', fn[1])

        # point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)   # original
        # point_set = np.load(fn[1]).astype(np.float32)                    # Bernardo
        
        try:
            if fn[1].endswith('.npy'):
                point_set = np.load(fn[1]).astype(np.float32)                  # Bernardo
            # elif fn[1].endswith('.ply'):
            #     point_set = self._readply(fn[1]).astype(np.float32)          # Bernardo
        except ValueError as ve:
            print(ve)
            print(f'fn[1]: \'{fn[1]}\'')
            sys.exit(0)

        # Bernardo
        if point_set.shape[1] == 7:        # if contains curvature
            point_set = point_set[:,:-1]   # remove curvature column

        point_set[:,0:3] = self.pc_normalize(point_set[:,0:3])
        '''
        if not self.normal_channel:
            point_set = point_set[:,0:3]
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls)
        '''
        return point_set, cls
    
    # def _readply(self, file):
    #     with open(file, 'rb') as f:
    #         plydata = PlyData.read(f)
    #         num_verts = plydata['vertex'].count
    #         vertices = np.zeros(shape=(num_verts, 3), dtype=np.float32)
    #         vertices[:,0] = plydata['vertex'].data['x']
    #         vertices[:,1] = plydata['vertex'].data['y']
    #         vertices[:,2] = plydata['vertex'].data['z']
    #         # vertices[:,3] = plydata['vertex'].data['red']
    #         # vertices[:,4] = plydata['vertex'].data['green']
    #         # vertices[:,5] = plydata['vertex'].data['blue']
    #         # print('plydata:', plydata)
    #         # print('vertices:', vertices)
    #         # sys.exit(0)
    #         return vertices  

    def pc_normalize(self,pc):
        # Bernardo
        pc /= 100
        pc = (pc - pc.min()) / (pc.max() - pc.min())

        # l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m

        return pc


    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """
