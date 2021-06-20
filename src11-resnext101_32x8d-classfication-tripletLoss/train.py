# -*- coding: utf-8 -*-
import multiprocessing
import time
import torch
import torch.nn as nn
from torchsummary import summary
import torch_optimizer as optim  # https://github.com/jettify/pytorch-optimizer

from cfg import _metrics, _fit, _modelcheckpoint, _reducelr, _criterion
# from gpu_selection import singleGPU, multiGPU, SingleModelCheckPoint, ParallelModelCheckpoint
# from utils import *
from data_gen import data_flow
from models.model import ResNet101, Efficient, Productnet

import os

def model_fn(args):
	# model = Productnet(backbone_weights=args.pretrained_weights, num_classes=args.num_classes)
	model = Productnet(backbone_weights=None, num_classes=args.num_classes)
	pretrained_dict = torch.load(args.pretrained_weights)
	single_dict = {}
	for k, v in pretrained_dict.items():
		single_dict[k[7:]] = v
	# single_dict.pop('arcface_loss.weight')
	model.load_state_dict(single_dict, strict=False)

	for param in model.parameters():
		param.requires_grad = True

	for name, value in model.named_parameters():
		if 'arcface_loss' in name:
			value.requires_grad = True
		if 'feature' in name:
			value.requires_grad = True
		if 'layer4' in name:
			value.requires_grad = True
		# if not name.startswith('layer'):
		# 	value.requires_grad = True

	for name, value in model.named_parameters():
		print(name, value.requires_grad)

	model = nn.DataParallel(model)
	model = model.cuda()

	return model

def train_model(args):
	model = model_fn(args)

	# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
	radam = optim.Ranger(params=model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
	optimizer = optim.Lookahead(radam)
	# optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

	criterion = {'Loss_Classification' : None, 'Loss_Distance' : None}
	# criterion = {'Loss_Distance' : None}
	metrics = {"acc@1" : _metrics.top1_accuracy, "acc@5" : _metrics.topk_accuracy}

	checkpoint1 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_Loss_Classification_{val_Loss_Classification:.4f}-val_Loss_Distance_{val_Loss_Distance:.4f}.pth'), monitor='val_Loss_Classification', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint2 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_Loss_Classification_{val_Loss_Classification:.4f}-val_Loss_Distance_{val_Loss_Distance:.4f}.pth'), monitor='val_Loss_Distance', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	# checkpoint1 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_Loss_Distance_{val_Loss_Distance:.4f}.pth'), monitor='val_Loss_Distance', mode='min', verbose=1, save_best_only=True, save_weights_only=True)

	# reduce_lr = _reducelr.StepLR(optimizer, factor=0.1, patience=10, min_lr=1e-8)
	reduce_lr = _reducelr.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-6)

	_fit.Fit(
			data_flow = data_flow,
			model=model,
			args=args,
			batch_size = args.batch_size,
			optimizer=optimizer,
			criterion=criterion,
			metrics=metrics,
			reduce_lr = reduce_lr,
			checkpoint = [checkpoint1, checkpoint2],
			verbose=1,
			workers=int(multiprocessing.cpu_count() * 0.8),
			# workers=1,
		)
	print('training done!')


