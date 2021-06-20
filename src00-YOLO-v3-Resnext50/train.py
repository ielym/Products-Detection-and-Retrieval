# -*- coding: utf-8 -*-
import multiprocessing
import time
import torch
import torch.nn as nn
from torchsummary import summary
import torch_optimizer as optim  # https://github.com/jettify/pytorch-optimizer
import os

from cfg import _metrics, _fit, _modelcheckpoint, _reducelr, _criterion
from data_gen import data_flow
from models.model import YOLO


def model_fn(args):

	yolo_params = {
		'mode' : args.mode,
		'num_anchors_per_level' : args.num_anchors_per_level,
		'bg_threshold' : args.bg_threshold,
		'num_classes' : args.num_classes,
		'data_local' : args.data_local,

		'prob_threshold' : args.prob_threshold,
	}

	model = YOLO(
					backbone_name=args.backbone_name,
					backbone_weights = args.backbone_weights,
					backbone_pretrained=args.backbone_pretrained,

					yolo_params = yolo_params,
				)

	parallel_dict = torch.load(args.pretrained_weights)
	single_dict = {}
	for k, v in parallel_dict.items():
		single_dict[k[7:]] = v
	model.load_state_dict(single_dict)

	for param in model.parameters():
		param.requires_grad = False

	model = nn.DataParallel(model)
	model = model.cuda()
	return model

def train_model(args):
	model = model_fn(args)


	# ========================================================  Pretrain  ==============================================
	for name, value in model.named_parameters():
		if 'detect_bottle' in name:
			value.requires_grad = True
		if 'predictor' in name:
			value.requires_grad = True
	for name, value in model.named_parameters():
		print(name, value.requires_grad)

	# args.iterations = 80 * 40
	args.iterations = 80 * 0
	args.learning_rate = 1e-4
	radam = optim.Ranger(params=model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
	optimizer = optim.Lookahead(radam)
	# optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
	# optimizer = optim.Ranger(params=model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
	# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
	criterion = {'Loss_Yolo' : None}
	checkpoint1 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'Iter-{iteration:05d}-val_Loss_Yolo_{val_Loss_Yolo:.4f}.pth'), monitor='val_Loss_Yolo', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint2 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'Iter-{iteration:05d}-val_Loss_Yolo_{val_Loss_Yolo:.4f}.pth'), monitor='val_Loss_Yolo', mode='min', verbose=1, save_best_only=False, save_weights_only=True)
	# reduce_lr = _reducelr.StepLR(optimizer, factor=0.1, patience=10000, min_lr=1e-8)
	# reduce_lr = _reducelr.CosineAnnealingWarmRestarts(optimizer, T_0=args.iterations, T_mult=1, eta_min=1e-8, last_epoch=-1)
	# reduce_lr = _reducelr.LinerWarmup(optimizer, start_lr=args.learning_rate, target_lr=1e-5, steps=args.iterations)
	reduce_lr = _reducelr.StepLRAfter(optimizer, steps=[9999999], factors=[0.1], min_lr=1e-8)
	_fit.Fit(
			data_flow = data_flow,
			model=model,
			args=args,
			optimizer=optimizer,
			criterion=criterion,
			metrics=None,
			reduce_lr = reduce_lr,
			checkpoint = [checkpoint1, checkpoint2],
			verbose=1,
			workers=int(multiprocessing.cpu_count() * 0.5),
			# workers=0,
		)
	print('Pretrain done!')

	# =======================================================  Train All  ==============================================
	for param in model.parameters():
		param.requires_grad = True
	for name, value in model.named_parameters():
		print(name, value.requires_grad)
	args.iterations = 160 * 100
	args.learning_rate = 1e-4
	radam = optim.Ranger(params=model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
	optimizer = optim.Lookahead(radam)
	criterion = {'Loss_Yolo' : None}
	checkpoint1 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'Iter-{iteration:05d}-val_Loss_Yolo_{val_Loss_Yolo:.4f}.pth'), monitor='val_Loss_Yolo', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint2 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'Iter-{iteration:05d}-val_Loss_Yolo_{val_Loss_Yolo:.4f}.pth'), monitor='val_Loss_Yolo', mode='min', verbose=1, save_best_only=False, save_weights_only=True)
	# reduce_lr = _reducelr.StepLRAfter(optimizer, steps=[5000, 7000], factors=[0.1, 0.1, 0.1], min_lr=1e-8)
	reduce_lr = _reducelr.CosineAnnealingWarmRestarts(optimizer, T_0=args.iterations, T_mult=1, eta_min=1e-8, last_epoch=-1)
	_fit.Fit(
			data_flow = data_flow,
			model=model,
			args=args,
			optimizer=optimizer,
			criterion=criterion,
			metrics=None,
			reduce_lr = reduce_lr,
			checkpoint = [checkpoint1, checkpoint2],
			verbose=1,
			workers=int(multiprocessing.cpu_count() * 0.5),
			# workers=0,
		)
	print('Training done!')
