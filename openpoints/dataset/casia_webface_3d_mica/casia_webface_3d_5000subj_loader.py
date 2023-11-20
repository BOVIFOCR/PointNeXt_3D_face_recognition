"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""

from ..build import DATASETS
from .casia_webface_3d_mica_loader import CASIA_WEBFACE_3D_MICA


@DATASETS.register_module()
class CASIA_WEBFACE_3D_MICA_5000subj(CASIA_WEBFACE_3D_MICA):

    def __init__(self,
                 num_points=2900,
                 data_dir='/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/output_5000class',
                 split='train',
                 transform=None
                 ):

        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        self.num_points = num_points
        self.transform = transform

        # 5000 CLASSES
        self.n_classes = 5000
        self.DATA_PATH = data_dir

        super().__init__(self.n_classes, self.num_points, self.DATA_PATH, split, transform)

