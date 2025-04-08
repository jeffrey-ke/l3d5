import numpy as np
from jutils.utils import pdb
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_seg

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=500, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--use_transform', action="store_true")
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(args).cuda()
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(args.num_points,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).cuda()
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data, with_smax=True)
    pred_label = torch.argmax(pred_label, dim=-1).cpu()

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    test_data = test_data.cpu()
    print("stats: ")
    print("% of dataset is chairs: ", len(test_label[test_label == 0])/len(test_label))
    print("% of dataset is vases: ", len(test_label[test_label == 1])/len(test_label))
    print("% of dataset is lamp: ", len(test_label[test_label == 2])/len(test_label))
    print(f"Predicted label on idx {args.i}: ", pred_label[args.i].long().item())
    print(f"GT label on idx {args.i}: ", test_label[args.i].long().item())
    viz_seg(test_data[args.i], torch.ones_like(test_data[args.i][..., 1]), "{}/{}points_idx{}.gif".format(args.output_dir, args.num_points, args.i), args.device, args.num_points)
    print ("test accuracy: {}".format(test_accuracy))

