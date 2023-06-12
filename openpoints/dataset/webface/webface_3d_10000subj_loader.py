"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""

from ..build import DATASETS
from .webface_3d_loader import WEBFACE_3D


@DATASETS.register_module()
class WEBFACE_3D_10000subj(WEBFACE_3D):

    def __init__(self,
                 num_points=2900,
                 # data_dir='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_1000subj',
                 data_dir='/nobackup1/bjgbiesseck/datasets/WebFace260M_3D_reconstruction_originalMICA/images_10000subj',  # peixoto
                 split='train',
                 transform=None
                 ):

        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        self.num_points = 2900
        self.transform = transform

        # 10000 CLASSES
        self.n_classes = 10000
        self.DATA_PATH = data_dir
        
        super().__init__(self.n_classes, self.num_points, self.DATA_PATH, split, transform)

