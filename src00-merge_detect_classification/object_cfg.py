# -*- coding: utf-8 -*-
import argparse
import os
import random
import numpy as np
import torch
import warnings
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('--mode', default='Test', choices=['Train', 'Test'], type=str, help='')
parser.add_argument('--model_name', default='Faster-Resnext101_32x8d', type=str, help='')

# Detection
parser.add_argument('--backbone_pretrained', default=False, type=bool, help='')
parser.add_argument('--backbone_name', default=r'resnext101_32x8d', type=str, help='vgg16, resnet50, resnext101_32x8d')
parser.add_argument('--backbone_weights', default=r'./model_classfication/zoo/resnext101_32x8d-8ba56ff5.pth', type=str, help='')

# RPN params
parser.add_argument('--num_anchor', default=9, type=int, help='')
parser.add_argument('--rpn_positive_iou_threshold', default=0.7, type=float, help='')
parser.add_argument('--rpn_negative_iou_threshold', default=0.3, type=float, help='')
parser.add_argument('--rpn_num_samples', default=256, type=int, help='')
parser.add_argument('--rpn_pn_fraction', default=0.5, type=float, help='')
parser.add_argument('--rpn_remove_min_size', default=1, type=float, help='')
parser.add_argument('--rpn_pre_nms_top_n_train', default=2000, type=int, help='')
parser.add_argument('--rpn_post_nms_top_n_train', default=1000, type=int, help='')
parser.add_argument('--rpn_nms_thresh', default=0.7, type=float, help='')

# Fast params
parser.add_argument('--roi_height', default=7, type=int, help='')
parser.add_argument('--roi_width', default=7, type=int, help='')
parser.add_argument('--fast_positive_iou_threshold', default=0.5, type=float, help='')
parser.add_argument('--fast_negative_iou_threshold', default=0.5, type=float, help='')
parser.add_argument('--fast_num_samples', default=64, type=int, help='')
parser.add_argument('--fast_pn_fraction', default=0.25, type=float, help='')
parser.add_argument('--fast_hidden', default=1024, type=int, help='')

# Test
parser.add_argument('--rpn_pre_nms_top_n_test', default=500, type=int, help='')
parser.add_argument('--rpn_post_nms_top_n_test', default=200, type=int, help='')
parser.add_argument('--remove_min_score', default=0.2, type=float, help='')
parser.add_argument('--remove_min_size', default=5, type=int, help='')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='')
parser.add_argument('--inference_weights', default='./model_classfication/ep00096-val_rpn_clss_Loss_0.0008-val_rpn_bbox_Loss_0.0009--val_fast_clss_Loss_0.0457-val_fast_bbox_Loss_0.0013.pth', type=str, help='')

# Data generation
parser.add_argument('--input_size', default=512, type=int, help='')
parser.add_argument('--num_classes', default=1, type=int, help='')
parser.add_argument('--max_objs', default=200, type=int, help='')

parser.add_argument('--seed', default=2021, type=int, help='0/1/2/... or None')
args, unknown = parser.parse_known_args()
