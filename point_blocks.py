import torch
from jutils.utils import pdb
import torch.nn as nn
import torch.nn.functional as F


class InputTransform(nn.Module):
    def __init__(self, args=None):
        super(InputTransform, self).__init__()
        N = args.n_points
        self.device = args.device
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,(1,1),1), # don't need any padding probably because it's a 1x1 kernel
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,(1,1),1),
            nn.ReLU(),
            nn.BatchNorm2d(128) # is it really batchnorm 2D? Across the batch,
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128,1024,(1,1),1),
            nn.ReLU(),
            nn.BatchNorm2d(1024) # is it really batchnorm 2D? Across the batch,
        )
        self.pointwise_mlps = nn.Sequential(
                self.block1,
                self.block2,
                self.block3,
                )
        self.maxpool2d = nn.MaxPool2d((N,1))
        self.fc512 = nn.Sequential(
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.BatchNorm1d(512)
        )
        self.fc256 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256)
        )
        self.to_transform = nn.Linear(256, 9)
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)
        self.to_transform.apply(init)


    def forward(self, X): # X is shape B,N,3
        B,N,*_ = X.shape
        X = X.unsqueeze(-1)
        X = X.permute((0,2,1,3)) # shape is B,3,N,1
        X = self.pointwise_mlps(X)
        X = self.maxpool2d(X) # collapses (B,1024,N,1) to (B,1024,1) probably
        X = X.view(B,-1)
        X = self.fc512(X)
        X = self.fc256(X)
        X = self.to_transform(X) + torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 1.], device=self.device)
        return X.view(B,3,3)

class PerPointMLP(nn.Module):
    def __init__(self, in_dim, out_dims, args=None):
        super(PerPointMLP, self).__init__()
        mlps = []
        last_dim = in_dim
        for d in out_dims:
            mlps.append(nn.Conv2d(last_dim, d, (1,1),1)),
            mlps.append(nn.ReLU()),
            mlps.append(nn.BatchNorm2d(d))
        self.perpoint_mlps = nn.Sequential(*mlps)

    def forward(self, X):
        mlped = self.perpoint_mlps(X)
        return mlped
