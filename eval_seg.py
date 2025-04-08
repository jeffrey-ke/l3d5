import numpy as np
from jutils.utils import pdb
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=500, help='The number of points per object to be included in the input data')
    parser.add_argument('--use_transform', action="store_true")

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(args).cuda()
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(args.num_points,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).cuda()
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind]).cuda()

    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data, with_smax=True)
    pred_label = torch.argmax(pred_label, dim=-1)

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    disagreement = (test_label != pred_label).float()
    thres = 0.30
    disagreement = torch.sum(disagreement, dim=-1) / disagreement.shape[1]
    very_disagreed = (disagreement > thres).nonzero().squeeze(-1)
    print(f"% of examples segmented with error > {thres}: ", len(very_disagreed) / len(disagreement))
    np.random.seed(10)
    print("Disagreeing idxs: ", np.random.choice(very_disagreed.cpu(), 5, replace=False))
    test_data = test_data.cpu()
    test_label = test_label.cpu()
    
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}_{}.gif".format(args.output_dir, args.num_points, args.i), args.device, args.num_points)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}_{}.gif".format(args.output_dir, args.num_points, args.i), args.device, args.num_points)
