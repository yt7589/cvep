# 系统配置信息类
import os
import pandas as pd

class Config(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")
        ###############################
        #### add dataset info here ####
        ###############################
        self.train_batch = args.train_batch
        self.val_batch = args.val_batch
        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno
        if args.dataset == 'CUB':
            self.dataset = args.dataset
            self.anno_root = '/home/zjkj/yantao/dcl/datasets/CUB_200_2011/anno'
            # 正式环境
            self.num_brands = 1500 # 249 # 206 # 品牌数
            self.num_bmys = 20000 # 3539 # 3421 # 年款数
            '''
            # 所里测试
            self.num_brands = 154
            self.num_bmys = 2458
            '''
        elif args.dataset == 'AIR': # 斯坦福AIR数据集处理方式
            self.dataset = args.dataset
            self.rawdata_root = './dataset/aircraft/data'
            self.anno_root = './dataset/aircraft/anno'
            self.numcls = 100
        else:
            raise Exception('dataset not defined ???')
        # 指定训练和测试数据集文件
        '''
        # 正式环境
        train_ds_file = 'fds_train_ds_20200926.txt'
        val_ds_file = 'wxs_ftds_20201005.txt'
        test_ds_file = 'wxs_ftds_20201005.txt'
        '''
        # 无锡所测试程序
        train_ds_file = 'bid_brand_train_ds_20200930.txt'
        val_ds_file = 'bid_brand_test_ds_20200925.txt'
        test_ds_file = 'bid_brand_test_ds_20200925.txt'
        if 'train' in get_list:
            self.train_anno = pd.read_csv(os.path.join(self.anno_root, train_ds_file),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
        if 'val' in get_list:
            # 正式环境：品牌为主任务
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, val_ds_file),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
        if 'test' in get_list:
            # 正式环境：品牌为主任务
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, test_ds_file),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
        self.swap_num = args.swap_num
        # 设置模型Checkpoint文件保存位置
        self.save_dir = './net_model/'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # 设置网络结构
        self.backbone = args.backbone
        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        # 当为True可以提升小样本类别精度
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False
        self.weighted_sample = False
        self.cls_2 = False
        self.cls_2xmul = True
        # 是否使用主类别控制子类别
        self.task1_control_task2 = False
        # 设置日志目录
        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)




