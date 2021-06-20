# -*- coding: utf-8 -*-
import argparse
import os
import random
import numpy as np
import torch
import warnings
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--mode', default='Train', choices=['Train', 'Test'], type=str, help='')
parser.add_argument('--model_name', default='Faster-Resnext101_32x8d', type=str, help='')
parser.add_argument('--data_local', default=r'/home/ymluo/DataSets/productsDET/V01', type=str, help='')
# parser.add_argument('--data_local', default=r'S:\DataSets\productsDET\V01', type=str, help='')

# Data generation
parser.add_argument('--input_size', default=512, type=int, help='')
parser.add_argument('--num_classes', default=1, type=int, help='')
parser.add_argument('--max_objs', default=200, type=int, help='')

# BackBone
parser.add_argument('--backbone_pretrained', default=True, type=bool, help='')
parser.add_argument('--backbone_name', default=r'resnext101_32x8d', type=str, help='vgg16, resnet50, resnext101_32x8d')
parser.add_argument('--backbone_weights', default=r'./models/zoo/resnext101_32x8d-8ba56ff5.pth', type=str, help='')

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

# Train
parser.add_argument('--start_epoch', default=0, type=int, help='')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='')
parser.add_argument('--max_epochs', default=5000, type=int, help='')
# parser.add_argument('--pretrained_weights', default='./models/ep00011-val_acc@1_62.5964-val_lossS_0.8536-val_lossL_0.9086.pth', type=str, help='')

# Test
parser.add_argument('--rpn_pre_nms_top_n_test', default=1000, type=int, help='')
parser.add_argument('--rpn_post_nms_top_n_test', default=500, type=int, help='')
parser.add_argument('--remove_min_score', default=0.1, type=float, help='')
parser.add_argument('--remove_min_size', default=5, type=int, help='')
parser.add_argument('--nms_thresh', default=0.2, type=float, help='')
parser.add_argument('--inference_weights', default='./models/ep00056-val_rpn_clss_Loss_0.0001-val_rpn_bbox_Loss_0.0001--val_fast_clss_Loss_0.0073-val_fast_bbox_Loss_0.0013.pth', type=str, help='')

parser.add_argument('--seed', default=2021, type=int, help='0/1/2/... or None')

args, unknown = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
print('CUDA device count : {}'.format(torch.cuda.device_count()))

def check_args(args):
	print(args.data_local)
	if not os.path.exists(args.data_local):
		raise Exception('FLAGS.data_local_path: %s is not exist' % args.data_local)

def set_random_seeds(args):
	os.environ['PYTHONHASHSEED'] = str(args.seed)
	cudnn.deterministic = True
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	print('You have chosen to seed training with seed {}.'.format(args.seed))

def main(args,**kwargs):
	# check_args(args)
	if args.seed != None:
		set_random_seeds(args)
	else:
		print('You have chosen to random seed.')
	if args.mode == 'Train':
		from train import train_model
		train_model(args=args)
	elif args.mode == 'Test':
		# from inference import Inference
		from submit import Inference
		Inference(args=args)

if __name__ == '__main__':
	main(args)
