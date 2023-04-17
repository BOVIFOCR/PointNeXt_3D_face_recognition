from __future__ import print_function

import sys
import os
import glob
from pathlib import Path

# BERNARDO
class TreeSyntheticFacesGPMM:
    def __walk(self, dir_path=Path()):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                # yield from self.__walk(path)
                yield self.__walk(path)

    def get_all_sub_folders(self, dir_path=''):
        # dir_path = dir_path.replace('//', '/')
        folders = [dir_path]
        for folder in self.__walk(Path(dir_path)):
            if isinstance(folder, str):
                folders.append(folder)
        return sorted(folders)

    def get_sub_folders_one_level(self, dir_path=''):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)

    def get_all_pointclouds_paths(self, dir_path, pc_ext='.bc'):
        all_sub_folders = self.get_all_sub_folders(dir_path)[1:]
        all_pc_paths = []
        for sub_folder_pointcloud in all_sub_folders:
            pc_paths = sorted(glob.glob(sub_folder_pointcloud + '/*' + pc_ext))
            all_pc_paths += pc_paths
        return all_pc_paths

    def get_pointclouds_paths(self, dir_path, num_classes=1000, num_expressions=50, pc_ext='.bc'):
        all_sub_folders = self.get_all_sub_folders(dir_path)[1:]
        pc_paths = [[]] * num_classes
        for i in range(num_classes):
            sub_folder_pointcloud = all_sub_folders[i]
            paths_one_folder = sorted(glob.glob(sub_folder_pointcloud + '/*' + pc_ext))
            pc_paths[i] = paths_one_folder[:num_expressions]
        return pc_paths

    def get_pointclouds_paths_with_subjects_names(self, dir_path, num_classes=1000, num_expressions=50, pc_ext='.bc'):
        assert num_classes > 0 and num_classes <= 10000 and num_expressions > 0 and num_expressions <= 50
        all_sub_folders = self.get_all_sub_folders(dir_path)[1:]
        pc_paths = [[]] * num_classes
        subjects_names = [[]] * num_classes
        unique_subjects_names = [''] * num_classes
        for i in range(num_classes):
            sub_folder_pointcloud = all_sub_folders[i]
            paths_one_folder = sorted(sorted(glob.glob(sub_folder_pointcloud + '/*' + pc_ext)))
            pc_paths[i] = paths_one_folder[:num_expressions]
            subject_name = paths_one_folder[0].split('/')[-2]
            subjects_names[i] = [subject_name] * num_expressions
            unique_subjects_names[i] = subject_name
        
        pc_subjects_paths = []
        for i in range(num_classes):
            for j in range(len(pc_paths[i])):
                pc_subjects_paths.append((subjects_names[i][j], pc_paths[i][j]))
        return pc_subjects_paths, unique_subjects_names




if __name__ == '__main__':
    synthetic_faces_path = '/home/bjgbiesseck/GitHub/3DFacePointCloudNet/Data/TrainData'

    # all_pc_paths = TreeSyntheticFacesGPMM().get_pointclouds_paths(dir_path=synthetic_faces_path, num_classes=100, num_expressions=5)
    # print('all_pc_paths:', all_pc_paths)
    
    pc_subjects_paths, unique_subjects_names = TreeSyntheticFacesGPMM().get_pointclouds_paths_with_subjects_names(dir_path=synthetic_faces_path, num_classes=10, num_expressions=5)
    print('pc_subjects_paths:', pc_subjects_paths)
    # for i in range(len(pc_subjects_paths)):
    #     print('pc_subjects_paths[i]:', pc_subjects_paths[i])
    print('unique_subjects_names:', unique_subjects_names)