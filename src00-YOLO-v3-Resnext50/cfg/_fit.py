import sys
sys.path.append("..")
sys.path.append(".")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
from ._history import History
from utils.mixmethods import mixup_data, mixup_criterion, snapmix_data, snapmix_criterion

def _register_history(criterion=None, metrics=None):
	history = History()

	if criterion:
		for k, v in criterion.items():
			name = k
			val_name= 'val_{}'.format(name)

			history.register(name, ':.4e')
			history.register(val_name, ':.4e')

	if metrics:
		for k, v in metrics.items():
			name = k
			val_name= 'val_{}'.format(name)

			history.register(name, ':6.3f')
			history.register(val_name, ':6.3f')

	return history

def Fit(data_flow, model, args, optimizer, criterion, metrics, reduce_lr, checkpoint, verbose, workers):

	history = _register_history(criterion, metrics)

	train_dataset, validation_dataset = data_flow(base_dir=args.data_local, input_size=args.input_size, max_objs=args.max_objs, num_classes=args.num_classes)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
	validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True)

	iteration = 1
	while iteration <= args.iterations:

		s_time = time.time()
		for images, targets in train_loader:
			model.train()
			# ==========================================================================================================
			images = images.cuda(non_blocking=True)
			targets = targets.cuda(non_blocking=True)

			_, yolo_loss = model(images, targets, iteration)

			optimizer.zero_grad()
			loss = torch.mean(yolo_loss['Loss_Yolo'])
			history.update(n=len(images), name='Loss_Yolo', val=loss.item())
			loss.backward()
			optimizer.step()
			# ==========================================================================================================

			history.update(n=1, name='Time', val=time.time()-s_time)
			if (iteration) % verbose == 0:
				history.display(iteration, optimizer.param_groups[0]['lr'], 'train')
			if (iteration) % args.save_model_step == 0:
				_Validate(model, history, validation_loader, iteration, criterion, metrics, args)
				for ckpt in checkpoint:
					ckpt.savemodel(model, iteration=iteration, **history.history_dict)
				history.reset()
			reduce_lr.step(history=history, iteration=iteration)
			iteration += 1
			s_time = time.time()

def _Validate(model, history, val_loader, iteration, criterion, metrics, args):
	model.eval()
	with torch.no_grad():
		for images, targets in val_loader:
			images = images.cuda(non_blocking=True)
			targets = targets.cuda(non_blocking=True)

			_, yolo_loss = model(images, targets, iteration)

			loss = torch.mean(yolo_loss['Loss_Yolo'])
			history.update(n=len(images), name='val_Loss_Yolo', val=loss.item())

		history.display(iteration=iteration, mode='validate')
