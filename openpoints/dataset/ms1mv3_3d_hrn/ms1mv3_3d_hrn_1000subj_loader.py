"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""

from ..build import DATASETS
from .ms1mv3_3d_hrn_loader import MS1MV3_3D_HRN


@DATASETS.register_module()
class MS1MV3_3D_HRN_1000subj(MS1MV3_3D_HRN):
    
    def __init__(self,
                 num_points=10000,
                 data_dir='/datasets1/bjgbiesseck/MS-Celeb-1M/ms1m-retinaface-t1/3D_reconstruction/HRN/images1000subj',
                 split='train',
                 transform=None
                 ):

        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        self.num_points = num_points
        self.transform = transform

        # 1000 CLASSES
        self.n_classes = 1000
        self.DATA_PATH = data_dir
        
        super().__init__(self.n_classes, self.num_points, self.DATA_PATH, split, transform)

