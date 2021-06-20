import sys
sys.path.append("..")
sys.path.append(".")
import torch
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
	train_dataset, validation_dataset = data_flow(base_dir=args.data_local, input_size=args.input_size, max_objs=args.max_objs, num_classes=args.num_classes)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
	validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True)

	for epoch in range(args.start_epoch, args.max_epochs):

		reduce_lr.step(history=history, epoch=epoch)
		print('\nEpoch {} learning_rate : {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

		_Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch, args)
		_Validate(model, history, validation_loader, criterion, metrics, epoch, args)

		for ckpt in checkpoint:
			ckpt.savemodel(model, epoch=epoch+1, **history.history_dict)


def _Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch, args):

	model.train()
	history.reset()
	s_time = time.time()
	total_batch = len(train_loader)

	for batch, (images, targets) in enumerate(train_loader):
		history.update(n=1, name='DataTime', val=time.time() - s_time)

		images = images.cuda(non_blocking=True)
		targets = targets.cuda(non_blocking=True)

		predicted, rpn_losses, fast_losses = model(images, targets)

		optimizer.zero_grad()
		rpn_clss_loss = torch.mean(rpn_losses['clss_loss'])
		history.update(n=len(images), name='rpn_clss_Loss', val=rpn_clss_loss.item())
		rpn_bbox_loss = torch.mean(rpn_losses['bbox_loss'])
		history.update(n=len(images), name='rpn_bbox_Loss', val=rpn_bbox_loss.item())
		fast_clss_loss = torch.mean(fast_losses['clss_loss'])
		history.update(n=len(images), name='fast_clss_Loss', val=fast_clss_loss.item())
		fast_bbox_loss = torch.mean(fast_losses['bbox_loss'])
		history.update(n=len(images), name='fast_bbox_Loss', val=fast_bbox_loss.item())

		loss = rpn_clss_loss + rpn_bbox_loss + fast_clss_loss + fast_bbox_loss
		loss.backward()
		# rpn_clss_loss.backward(retain_graph=True)
		# rpn_bbox_loss.backward(retain_graph=True)
		# fast_clss_loss.backward(retain_graph=True)
		# fast_bbox_loss.backward()
		optimizer.step()

		history.update(n=1, name='BatchTime', val=time.time() - s_time)
		if (batch) % verbose == 0:
			history.display(epoch+1, batch+1, total_batch, 'train')
		s_time = time.time()

def _Validate(model, history, val_loader, criterion, metrics, epoch, args):
	model.eval()
	with torch.no_grad():
		for batch, (images, targets) in enumerate(val_loader):
			images = images.cuda(non_blocking=True)
			targets = targets.cuda(non_blocking=True)

			predicted, rpn_losses, fast_losses = model(images, targets)

			rpn_clss_loss = torch.mean(rpn_losses['clss_loss'])
			history.update(n=len(images), name='val_rpn_clss_Loss', val=rpn_clss_loss.item())
			rpn_bbox_loss = torch.mean(rpn_losses['bbox_loss'])
			history.update(n=len(images), name='val_rpn_bbox_Loss', val=rpn_bbox_loss.item())
			fast_clss_loss = torch.mean(fast_losses['clss_loss'])
			history.update(n=len(images), name='val_fast_clss_Loss', val=fast_clss_loss.item())
			fast_bbox_loss = torch.mean(fast_losses['bbox_loss'])
			history.update(n=len(images), name='val_fast_bbox_Loss', val=fast_bbox_loss.item())

		history.display(epoch=epoch+1, mode='validate')