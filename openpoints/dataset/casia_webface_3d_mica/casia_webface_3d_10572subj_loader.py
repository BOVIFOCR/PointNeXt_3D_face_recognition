"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""

from ..build import DATASETS
from .casia_webface_3d_mica_loader import CASIA_WEBFACE_3D_MICA


@DATASETS.register_module()
class CASIA_WEBFACE_3D_MICA_10572subj(CASIA_WEBFACE_3D_MICA):
    
    def __init__(self,
                 num_points=2900,
                 data_dir='/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/output',
                 split='train',
                 transform=None
                 ):

        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        self.num_points = 2900
        self.transform = transform

        # 10572 CLASSES
        self.n_classes = 10572
        self.DATA_PATH = data_dir
        
        super().__init__(self.n_classes, self.num_points, self.DATA_PATH, split, transform)

