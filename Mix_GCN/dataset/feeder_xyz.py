import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import tools

coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]

class Feeder(Dataset):
    def __init__(self, data_path: str, data_split: str, p_interval: list=[0.95], window_size: int=64, bone: bool=False, vel: bool=False):
        super(Feeder, self).__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.load_data()
        
    def load_data(self):
        if self.data_split == 'train':
          self.data = np.load(self.data_path + '/train_joint_bone.npy')  # 加载训练数据############################
          self.label = np.load(self.data_path + '/train_label.npy')  # 加载训练标签
          self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.data_split == 'test':
          self.data = np.load(self.data_path + '/test_A_joint_bone.npy')  # 加载测试数据#############################
          self.label = np.load(self.data_path + '/test_A_label.npy')  # 加载测试标签
          # self.data = np.load(self.data_path + '/test_bone.npy')  # 加载测试数据
          # self.label = np.load(self.data_path + '/train_label.npy')  # 加载测试标签
          self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        data_numpy = self.data[idx] # M T V C
        label = self.label[idx]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if(valid_frame_num == 0): 
            #return np.zeros((3, 64, 17, 2)), label, idx
            return np.zeros((6, 64, 17, 2)), label, idx
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1)) # all_joint - 0_joint
        return data_numpy, label, idx # C T V M
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
  
    
            
        
    