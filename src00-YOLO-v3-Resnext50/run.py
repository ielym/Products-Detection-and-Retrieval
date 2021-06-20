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
parser.add_argument('--model_name', default='YOLO3-Resnext101_32x8d', type=str, help='')
parser.add_argument('--data_local', default=r'/home/ymluo/DataSets/productsDET/V01', type=str, help='')
# parser.add_argument('--data_local', default=r'S:\DataSets\productsDET\V01', type=str, help='')

# Data generation
parser.add_argument('--input_size', default=608, type=int, help='')
parser.add_argument('--max_objs', default=200, type=int, help='')
parser.add_argument('--num_classes', default=116, type=int, help='')
parser.add_argument('--num_anchors_per_level', default=[3, 3, 3], type=list, help='')

# BackBone
parser.add_argument('--backbone_pretrained', default=True, type=bool, help='')
parser.add_argument('--backbone_name', default=r'resnet50', type=str, help='vgg16, darknet19, resnet50, resnext101_32x8d, efficientnet-b7')
parser.add_argument('--backbone_weights', default=r'./models/zoo/resnet50-19c8e357.pth', type=str, help='')
parser.add_argument('--feature_strides_levels', default=[8, 16, 32], type=int, help='')

# YoLO
parser.add_argument('--bg_threshold', default=0.5, type=float, help='')

# Train
parser.add_argument('--iterations', default=30000, type=int, help='')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='')
parser.add_argument('--save_model_step', default=40, type=int, help='')
parser.add_argument('--pretrained_weights', default='./models/Iter-03200-val_Loss_Yolo_90.0332.pth', type=str, help='')

# Test
parser.add_argument('--prob_threshold', default=0.2, type=float, help='')
parser.add_argument('--nms_threshold', default=0.2, type=float, help='')
parser.add_argument('--min_size', default=5, type=int, help='')
parser.add_argument('--inference_weights', default='./models/Iter-00576-val_Loss_Yolo_32.2315.pth', type=str, help='')

parser.add_argument('--seed', default=2021, type=int, help='0/1/2/... or None')

args, unknown = parser.parse_known_args()
def check_args(args):
	print(args.data_local)
	if args.mode=='Train' and not os.path.exists(args.data_local):
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
		os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
		print('CUDA device count : {}'.format(torch.cuda.device_count()))
		train_model(args=args)
	elif args.mode == 'Test':
		os.environ["CUDA_VISIBLE_DEVICES"] = '0'
		print('CUDA device count : {}'.format(torch.cuda.device_count()))
		from inference import Inference
		# from inference_video import Inference
		Inference(args=args)

if __name__ == '__main__':
	main(args)
