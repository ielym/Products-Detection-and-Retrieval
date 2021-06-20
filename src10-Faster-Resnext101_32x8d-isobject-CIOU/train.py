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
from models.model import FasterRcnn


def model_fn(args):

	model = FasterRcnn(args = args)

	for param in model.parameters():
		param.requires_grad = True

	model = nn.DataParallel(model)
	model = model.cuda()
	return model

def train_model(args):
	model = model_fn(args)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	criterion = {'rpn_clss_Loss' : None, 'rpn_bbox_Loss' : None, 'fast_clss_Loss' : None, 'fast_bbox_Loss' : None}
	metrics = {"acc" : _metrics.accuracy}

	checkpoint1 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_rpn_clss_Loss_{val_rpn_clss_Loss:.4f}-val_rpn_bbox_Loss_{val_rpn_bbox_Loss:.4f}--val_fast_clss_Loss_{val_fast_clss_Loss:.4f}-val_fast_bbox_Loss_{val_fast_bbox_Loss:.4f}.pth'), monitor='val_rpn_clss_Loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint2 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_rpn_clss_Loss_{val_rpn_clss_Loss:.4f}-val_rpn_bbox_Loss_{val_rpn_bbox_Loss:.4f}--val_fast_clss_Loss_{val_fast_clss_Loss:.4f}-val_fast_bbox_Loss_{val_fast_bbox_Loss:.4f}.pth'), monitor='val_rpn_bbox_Loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint3 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_rpn_clss_Loss_{val_rpn_clss_Loss:.4f}-val_rpn_bbox_Loss_{val_rpn_bbox_Loss:.4f}--val_fast_clss_Loss_{val_fast_clss_Loss:.4f}-val_fast_bbox_Loss_{val_fast_bbox_Loss:.4f}.pth'), monitor='val_fast_clss_Loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint4 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_rpn_clss_Loss_{val_rpn_clss_Loss:.4f}-val_rpn_bbox_Loss_{val_rpn_bbox_Loss:.4f}--val_fast_clss_Loss_{val_fast_clss_Loss:.4f}-val_fast_bbox_Loss_{val_fast_bbox_Loss:.4f}.pth'), monitor='val_fast_bbox_Loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)

	# reduce_lr = _reducelr.StepLR(optimizer, factor=0.1, patience=10, min_lr=1e-6)
	reduce_lr = _reducelr.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=1e-6)

	_fit.Fit(
			data_flow = data_flow,
			model=model,
			args=args,
			optimizer=optimizer,
			criterion=criterion,
			metrics=metrics,
			reduce_lr = reduce_lr,
			checkpoint = [checkpoint1, checkpoint2, checkpoint3, checkpoint4],
			verbose=1,
			workers=int(multiprocessing.cpu_count() * 0.8),
			# workers=0,
		)
	print('training done!')


