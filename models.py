import torch
import torch.nn as nn
import torch.nn.functional as F
from point_blocks import InputTransform, PerPointMLP


# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, args, num_classes=3):
        super(cls_model, self).__init__()
        N = args.n_points
        self.t_net = InputTransform(args)
        self.feature_net = InputTransform(args, dim=64) #kind of important
        self.mlp64_128_1024 = PerPointMLP(64,128,1024)
        self.maxpool = nn.MaxPool2d((N, 1))
        self.mlp512_256_k = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.DropOut(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.DropOut(p=0.3),
                nn.Linear(256, num_classes), # omitting softmax to use logits loss
                )


    def forward(self, points, with_smax=False):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        input transform
        linear 64 64
        feature transform
        linear 64 128 1024
        max pool into a 1024 (N,1024)
        linear 512 256 num_classes
        softmax probably. 

        pass



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        pass

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        pass



