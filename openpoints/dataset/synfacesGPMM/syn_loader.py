"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""

import os
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS
import sys

from data_loader.loader_synthetic_faces_gpmm.tree_synthetic_faces import TreeSyntheticFacesGPMM 
import struct

'''
def download_and_extract_archive(url, path, md5=None):
    # Works when the SSL certificate is expired for the link
    path = Path(path)
    extract_path = path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / Path(url).name
        if not file_path.exists() or not check_integrity(file_path, md5):
            print(f'{file_path} not found or corrupted')
            print(f'downloading from {url}')
            context = ssl.SSLContext()
            with urllib.request.urlopen(url, context=context) as response:
                with tqdm(total=response.length) as pbar:
                    with open(file_path, 'wb') as file:
                        chunk_size = 1024
                        chunks = iter(lambda: response.read(chunk_size), '')
                        for chunk in chunks:
                            if not chunk:
                                break
                            pbar.update(chunk_size)
                            file.write(chunk)
            extract_archive(str(file_path), str(extract_path))
    return extract_path


def load_data(data_dir, partition, url):
    download_and_extract_archive(url, data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    return all_data, all_label
'''

@DATASETS.register_module()
class synfacesGPMM(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.
    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    
    dir_name = '/datasets1/'
    md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
    url = f'https://shapenet.cs.stanford.edu/media/{dir_name}.zip'
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain',
               'desk',
               'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']

    def __init__(self,
                 num_points=1024,
                 data_dir="./data/ModelNet40Ply2048",
                 split='train',
                 transform=None
                 ):
        for i in range(20):
            print("Initializing SystheticFaces Net")
        
        
        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        # Load paths
        #self.data, self.label = load_data(data_dir, self.partition, self.url)
        #self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform
        
        #Load synfaces path
        DATA_PATH = '/home/pbqv20/datasets/SyntheticFacesGPMM'
        DATA_PATH += '/' + self.partition
        NUM_POINT = 28588
        self.num_points = NUM_POINT
        n_classes = 100
        n_expressions = 10
        self.data, unique_subjects_names = TreeSyntheticFacesGPMM().get_pointclouds_paths_with_subjects_names(DATA_PATH, num_classes=n_classes, num_expressions=n_expressions)
        print(len(self.data))
        
        
        self.cat = unique_subjects_names    # Bernardo
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        #self.num_classes = len(unique_subjects_names)
        #print(self.classes)
        #sys.exit(0)
        
    def __getitem__(self, item):
        pointcloud, cls = self._get_item(item)
        #print("_main get item_")
        #print(pointcloud.shape)
        #print(cls)
        
        ''' Old code
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        '''
        
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
        size = self.data
        size = len(size)
        return size

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
        point_set = self._readbcn(fn[1]).astype(np.float32)                 # Bernardo

        # Bernardo
        if point_set.shape[1] == 7:        # if contains curvature
            point_set = point_set[:,:-1]   # remove curvature column

        # Take the first npoints
        point_set = point_set[0:self.num_points,:]
        
       
        point_set[:,0:3] = self.pc_normalize(point_set[:,0:3])
        '''
        if not self.normal_channel:
            point_set = point_set[:,0:3]
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls)
        '''
        return point_set, cls
    
    def _readbcn(self, file):
        npoints = os.path.getsize(file) // 4
        with open(file,'rb') as f:
            raw_data = struct.unpack('f'*npoints,f.read(npoints*4))
            data = np.asarray(raw_data,dtype=np.float32)       
        # data = data.reshape(7, len(data)//7)   # original
        data = data.reshape(3, len(data)//3).T   # Bernardo
        return data                        # Bernardo    


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
