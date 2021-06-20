import sys
sys.path.append("..")
sys.path.append(".")
import torch
import torch.nn as nn
import time
from ._history import History
from utils.mixmethods import mixup_data, mixup_criterion, snapmix_data, snapmix_criterion

def _register_history(criterion, metrics):
	history = History()

	for k, v in criterion.items():
		name = k
		val_name= 'val_{}'.format(name)

		history.register(name, ':.4e')
		history.register(val_name, ':.4e')

	for k, v in metrics.items():
		name = k
		val_name= 'val_{}'.format(name)

		history.register(name, ':6.3f')
		history.register(val_name, ':6.3f')

	return history

def Fit(data_flow, model, args, optimizer, criterion, metrics, reduce_lr, checkpoint, verbose, workers):

	history = _register_history(criterion, metrics)
	for epoch in range(args.start_epoch, args.max_epochs):

		# reduce_lr.step(history=history, epoch=epoch)
		# print('\nEpoch {} learning_rate : {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

		train_dataset, validation_dataset = data_flow(base_dir=args.data_local, input_size=args.input_size, num_classes=args.num_classes)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True, collate_fn=train_dataset.collate_fn)
		validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=validation_dataset.collate_fn)

		_Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch, args, reduce_lr)
		_Validate(model, history, validation_loader, criterion, metrics, epoch, args)

		for ckpt in checkpoint:
			ckpt.savemodel(model, epoch=epoch+1, **history.history_dict)

def _Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch, args, reduce_lr):

	model.train()
	history.reset()
	s_time = time.time()
	total_batch = len(train_loader)

	for batch, (images, targets, masks) in enumerate(train_loader):
		history.update(n=1, name='DataTime', val=time.time() - s_time)

		reduce_lr.step(history=history, epoch=epoch*total_batch + batch)
		print('\nIter {} learning_rate : {}'.format(epoch*total_batch+batch+1, optimizer.param_groups[0]['lr']))

		images = images.cuda(non_blocking=True)
		targets = targets.cuda(non_blocking=True)
		masks = masks.cuda(non_blocking=True)

		_, yolo_loss = model(images, epoch, targets, masks)

		optimizer.zero_grad()
		loss = torch.mean(yolo_loss['Loss_Yolo'])
		history.update(n=len(images), name='Loss_Yolo', val=loss.item())
		loss.backward()
		# nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
		# nn.utils.clip_grad_value_(model.module.parameters(), 1)
		optimizer.step()

		history.update(n=1, name='BatchTime', val=time.time() - s_time)
		if (batch) % verbose == 0:
			history.display(epoch+1, batch+1, total_batch, 'train')
		s_time = time.time()

def _Validate(model, history, val_loader, criterion, metrics, epoch, args):
	model.eval()
	with torch.no_grad():
		for batch, (images, targets, masks) in enumerate(val_loader):
			images = images.cuda(non_blocking=True)
			targets = targets.cuda(non_blocking=True)
			masks = masks.cuda(non_blocking=True)

			_, yolo_loss = model(images, epoch, targets, masks)

			loss = torch.mean(yolo_loss['Loss_Yolo'])
			history.update(n=len(images), name='val_Loss_Yolo', val=loss.item())
		history.display(epoch=epoch+1, mode='validate')