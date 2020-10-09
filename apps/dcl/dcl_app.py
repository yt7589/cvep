# DCL模型应用系统
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from apps.dcl.conf.config import Config
from apps.dcl.transforms.transform_manager import TransformManager
from apps.dcl.ds.stvr_dataset import StvrDataset

class DclApp(object):
    def __init__(self):
        self.refl = 'apps.dcl.DclApp'

    def startup(self, args={}):
        print('DCL应用系统 v0.0.4')
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        args = self.parse_args()
        config = Config(args, 'train')
        config.cls_2 = args.cls_2
        config.cls_2xmul = args.cls_mul
        assert config.cls_2 ^ config.cls_2xmul
        transformers = TransformManager.load_data_transformers(
            args.resize_resolution, args.crop_resolution, 
            args.swap_num
        )
        # inital dataloader
        train_set = StvrDataset(Config = config,\
                            anno = config.train_anno,\
                            common_aug = transformers["common_aug"],\
                            swap = transformers["swap"],\
                            swap_size=args.swap_num, \
                            totensor = transformers["train_totensor"],\
                            train = True)
        trainval_set = StvrDataset(Config = config,\
                            anno = config.val_anno,\
                            common_aug = transformers["None"],\
                            swap = transformers["None"],\
                            swap_size=args.swap_num, \
                            totensor = transformers["val_totensor"],\
                            train = False,
                            train_val = True)
        val_set = StvrDataset(Config = config,\
                          anno = config.val_anno,\
                          common_aug = transformers["None"],\
                          swap = transformers["None"],\
                            swap_size=args.swap_num, \
                          totensor = transformers["test_totensor"],\
                          test=True)
        dataloader = {}
        dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                    batch_size=args.train_batch,\
                    shuffle=True,\
                    num_workers=args.train_num_workers,\
                    collate_fn=StvrDataset.collate_fn4train \
                        if not config.use_backbone \
                        else StvrDataset.collate_fn4backbone,
                    drop_last=True if config.use_backbone else False,
                    pin_memory=True)
        setattr(dataloader['train'], 'total_item_len', len(train_set))
        dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set,\
                    batch_size=args.val_batch,\
                    shuffle=False,\
                    num_workers=args.val_num_workers,\
                    collate_fn=StvrDataset.collate_fn4val \
                        if not config.use_backbone \
                        else StvrDataset.collate_fn4backbone,
                    drop_last=True if config.use_backbone else False,
                    pin_memory=True)
        setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))
        setattr(dataloader['trainval'], 'num_cls', config.num_brands)
        dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                    batch_size=args.val_batch,\
                    shuffle=False,\
                    num_workers=args.val_num_workers,\
                    collate_fn=StvrDataset.collate_fn4test \
                        if not config.use_backbone \
                        else StvrDataset.collate_fn4backbone,
                    drop_last=True if config.use_backbone else False,
                    pin_memory=True)
        setattr(dataloader['val'], 'total_item_len', len(val_set))
        setattr(dataloader['val'], 'num_cls', config.num_brands)
        cudnn.benchmark = True
        print('Choose model and train set', flush=True)





        
    def auto_load_resume(self, load_dir):
        folders = os.listdir(load_dir)
        #date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
        date_list = [int(x.split('_')[2]) for x in folders]
        choosed = folders[date_list.index(max(date_list))]
        weight_list = os.listdir(os.path.join(load_dir, choosed))
        acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
        acc_list = [float(x) for x in acc_list]
        choosed_w = weight_list[acc_list.index(max(acc_list))]
        return os.path.join(load_dir, choosed, choosed_w)
        
    # parameters setting
    def parse_args(self):
        parser = argparse.ArgumentParser(description='dcl parameters')
        parser.add_argument('--data', dest='dataset',
                            default='CUB', type=str)
        parser.add_argument('--epoch', dest='epoch',
                            default=360, type=int)
        parser.add_argument('--backbone', dest='backbone',
                            default='resnet50', type=str)
        parser.add_argument('--cp', dest='check_point',
                            default=1000, type=int)
        parser.add_argument('--sp', dest='save_point',
                            default=1000, type=int)
        parser.add_argument('--tb', dest='train_batch',
                            default=128, type=int)
        parser.add_argument('--tnw', dest='train_num_workers',
                            default=16, type=int)
        parser.add_argument('--vb', dest='val_batch',
                            default=256, type=int)
        parser.add_argument('--vnw', dest='val_num_workers',
                            default=16, type=int)
        parser.add_argument('--lr', dest='base_lr',
                            default=0.0008, type=float)
        parser.add_argument('--lr_step', dest='decay_step',
                            default=50, type=int)
        parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                            default=10.0, type=float)
        parser.add_argument('--start_epoch', dest='start_epoch',
                            default=0,  type=int)
        parser.add_argument('--detail', dest='cam',
                            default='cam', type=str)
        parser.add_argument('--size', dest='resize_resolution',
                            default=224, type=int)
        parser.add_argument('--crop', dest='crop_resolution',
                            default=224, type=int)
        parser.add_argument('--cls_2', dest='cls_2', default=True, 
                            action='store_true')
        parser.add_argument('--swap_num', default=[7, 7],
                        nargs=2, metavar=('swap1', 'swap2'),
                        type=int, help='specify a range')
        parser.add_argument('--auto_resume', dest='auto_resume',
                            default=False,
                            action='store_true')
                            
        parser.add_argument('--save', dest='resume',
                            default=None,
                            type=str)
        parser.add_argument('--cls_mul', dest='cls_mul',
                            action='store_true')
        args = parser.parse_args()
        return args