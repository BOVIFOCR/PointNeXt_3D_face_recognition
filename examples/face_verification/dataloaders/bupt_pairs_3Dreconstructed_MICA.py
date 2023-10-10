from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path

from .tree_bupt_3Dreconstructed_MICA import TreeBUPT_3DReconstructedMICA

# BERNARDO
class BUPT_Pairs_3DReconstructedMICA:

    def load_pointclouds_pairs_with_labels(self, root, protocol_file_path, file_ext='mesh_centralized_nosetip_croped_radius=100.npy'):
        all_pairs_paths_label, folds_indexes, pos_pair_label, neg_pair_label = TreeBUPT_3DReconstructedMICA().load_all_pairs_samples_from_protocol_file(root, protocol_file_path, file_ext)
        # print('\nLFW_Pairs_3DReconstructedMICA - load_pointclouds_pairs_with_labels')
        # print('all_pairs_paths_label:', all_pairs_paths_label)
        # print('len(all_pairs_paths_label):', len(all_pairs_paths_label))
        # print('pos_pair_label:', pos_pair_label)
        # print('neg_pair_label:', neg_pair_label)
        return all_pairs_paths_label, folds_indexes, pos_pair_label, neg_pair_label
