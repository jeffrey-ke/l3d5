import torch
import torch.nn as nn
import torch.nn.functional as F
from point_blocks import InputTransform, PerPointMLP


# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, args, num_classes=3):
        super(cls_model, self).__init__()
        N = args.n_points
        self.as_seg = args.task == "seg"
        self.t_net = InputTransform(args)
        self.mlp64_64 = PerPointMLP(in_dim=3, out_dims=[64,64])
        self.feature_transform = InputTransform(args, dim=64) #kind of important
        self.mlp64_128_1024 = PerPointMLP(in_dim=64, out_dims=[64,128,1024])
        self.maxpool = nn.MaxPool2d((N, 1))
        self.mlp512_256_k = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes), # omitting softmax to use logits loss
                )


    def forward(self, points, with_smax=False):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # how do I need to tranpose this business?
        transformed_points = points @ self.t_net(points).transpose(2,1) # What order should it be?
        transformed_points = transformed_points.unsqueeze(-1).permute(0,2,1,3)
        n64 = self.mlp64_64(transformed_points) # at this point we're at shape b,64,n,1
        # we want shape b,n,64
        n64 = n64.permute(0,2,1,3).squeeze(-1) #now shape b,n,64
        transformed_features = n64 @ self.feature_transform(n64).transpose(2,1) #shape b,64,64
        transformed_features = transformed_features.unsqueeze(-1).permute(0,2,1,3)
        # shape b,n,64,1 -> b,64,n,1
        n1024 = self.mlp64_128_1024(transformed_features) # shape b,1024,n,1
        global_features = self.maxpool(n1024) # shape b,1024,1 -> b,1024,
        global_features = global_features.view(*global_features.shape[:-2])
        if self.as_seg:
            _, n, *_ = n64.shape #n64 is shape b,n,64
            # initially global_features is shape b,1024
            # after unsqueeze it is b,1,1024, and after expanding it's b,n,1024
            global_features = global_features.unsqueeze(1).expand(-1, n, -1) 
            n1088 = torch.cat((n64, global_features), dim=-1) #b,n,1088
            return n1088
        else:
            output_scores = self.mlp512_256_k(global_features)
            if with_smax:
                return F.softmax(output_scores, dim=1)
            else:
                return output_scores




# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, args, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.cls_model = cls_model(args)
        m = args.num_seg_classes
        self.mlps = PerPointMLP(in_dim=1088, out_dims=[512, 256, 128, 128, m])
        # wondering if there's a reason for the separation here.

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        n1088 = self.cls_model(points)




