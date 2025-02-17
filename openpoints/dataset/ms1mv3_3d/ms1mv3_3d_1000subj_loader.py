"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""

from ..build import DATASETS
from .ms1mv3_3d_loader import MS1MV3_3D


@DATASETS.register_module()
class MS1MV3_3D_1000subj(MS1MV3_3D):
    
    def __init__(self,
                 num_points=2900,
                 data_dir='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_1000subj',
                 split='train',
                 transform=None
                 ):

        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        self.num_points = 2900
        self.transform = transform

        # 1000 CLASSES
        self.n_classes = 1000
        self.DATA_PATH = data_dir
        
        super().__init__(self.n_classes, self.num_points, self.DATA_PATH, split, transform)

