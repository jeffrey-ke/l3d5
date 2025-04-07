import argparse
import torch

from point_blocks import InputTransform

parser = argparse.ArgumentParser()
parser.add_argument("--n_points", type=int, default=1024, help="The number of points in each point cloud")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
a = InputTransform(args, dim=64).to("cuda")
input = torch.randn((100,1024,3), device="cuda")

from models import cls_model

m = cls_model(args).to("cuda")
a = m(input, with_smax=True)
print(a.shape)
