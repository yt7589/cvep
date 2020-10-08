#
from apps.dcl.transforms.normalize import Normalize
from apps.dcl.transforms.compose import Compose
from apps.dcl.transforms.random_swap import Randomswap
from apps.dcl.transforms.resize import Resize
from apps.dcl.transforms.random_rotation import RandomRotation
from apps.dcl.transforms.random_crop import RandomCrop
from apps.dcl.transforms.random_horizontal_flip import RandomHorizontalFlip
from apps.dcl.transforms.to_tensor import ToTensor
from apps.dcl.transforms.center_crop import CenterCrop

class TransformManager(object):
    @staticmethod
    def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
        center_resize = 600
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        data_transforms = {
            'swap': Compose([
                Randomswap((swap_num[0], swap_num[1])),
            ]),
            'common_aug': Compose([
                Resize((resize_reso, resize_reso)),
                RandomRotation(degrees=15),
                RandomCrop((crop_reso,crop_reso)),
                RandomHorizontalFlip(),
            ]),
            'train_totensor': Compose([
                Resize((crop_reso, crop_reso)),
                # ImageNetPolicy(),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val_totensor': Compose([
                Resize((crop_reso, crop_reso)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'test_totensor': Compose([
                Resize((resize_reso, resize_reso)),
                CenterCrop((crop_reso, crop_reso)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'None': None,
        }
        return data_transforms