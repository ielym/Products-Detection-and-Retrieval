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

def Fit(data_flow, model, args, batch_size, optimizer, criterion, metrics, reduce_lr, checkpoint, verbose, workers):

	history = _register_history(criterion, metrics)

	train_dataset, validation_dataset = data_flow(base_dir=args.data_local, input_size=args.input_size)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
	validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

	for epoch in range(args.start_epoch, args.max_epochs):

		reduce_lr.step(history=history, epoch=epoch)
		print('\nEpoch {} learning_rate : {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

		_Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch)
		_Validate(model, history, validation_loader, criterion, metrics, epoch)

		for ckpt in checkpoint:
			ckpt.savemodel(model, epoch=epoch+1, **history.history_dict)


def _Train(model, history, train_loader, optimizer, criterion, metrics, verbose, epoch):

	model.train()

	history.reset()
	s_time = time.time()
	total_batch = len(train_loader)

	# for batch, (img_anchor, img_same, img_differ, label_anchor, label_same, label_differ) in enumerate(train_loader):
	for batch, (img_anchor, label_anchor) in enumerate(train_loader):
		history.update(n=1, name='DataTime', val=time.time() - s_time)

		img_anchor = img_anchor.cuda(non_blocking=True)
		# img_same = img_same.cuda(non_blocking=True)
		# img_differ = img_differ.cuda(non_blocking=True)
		label_anchor = label_anchor.cuda(non_blocking=True)
		# label_same = label_same.cuda(non_blocking=True)
		# label_differ = label_differ.cuda(non_blocking=True)

		# loss_classification, loss_compare = model(img_anchor, img_same, img_differ, label_anchor, label_same, label_differ)
		# y_pre, loss_classification = model(img_anchor, img_same, img_differ, label_anchor, label_same, label_differ)
		y_pre, loss_classification = model(img_anchor, label_anchor)

		optimizer.zero_grad()
		metric1 = metrics['acc@1'](y_pre, label_anchor.long())
		history.update(n=img_anchor.shape[0], name='acc@1', val=metric1.item())
		metric5 = metrics['acc@5'](y_pre, label_anchor.long(), topk=5)
		history.update(n=img_anchor.shape[0], name='acc@5', val=metric5.item())

		loss_classification = torch.mean(loss_classification)
		history.update(n=img_anchor.shape[0], name='Loss_Classification', val=loss_classification.item())
		# loss_compare = torch.mean(loss_compare)
		# history.update(n=img_anchor.shape[0], name='Loss_Distance', val=loss_compare.item())

		# loss = loss_classification + loss_compare
		loss = loss_classification
		loss.backward()
		optimizer.step()

		history.update(n=1, name='BatchTime', val=time.time() - s_time)

		if (batch) % verbose == 0:
			history.display(epoch+1, batch+1, total_batch, 'train')
		s_time = time.time()

def _Validate(model, history, val_loader, criterion, metrics, epoch):
	model.eval()
	with torch.no_grad():
		# for batch, (img_anchor, img_same, img_differ, label_anchor, label_same, label_differ) in enumerate(val_loader):
		for batch, (img_anchor, label_anchor) in enumerate(val_loader):
			img_anchor = img_anchor.cuda(non_blocking=True)
			# img_same = img_same.cuda(non_blocking=True)
			# img_differ = img_differ.cuda(non_blocking=True)
			label_anchor = label_anchor.cuda(non_blocking=True)
			# label_same = label_same.cuda(non_blocking=True)
			# label_differ = label_differ.cuda(non_blocking=True)

			# y_pre, loss_classification = model(img_anchor, img_same, img_differ, label_anchor, label_same, label_differ)
			# loss_compare = model(img_anchor, img_same, img_differ, label_anchor, label_same, label_differ)
			y_pre, loss_classification = model(img_anchor, label_anchor)

			metric1 = metrics['acc@1'](y_pre, label_anchor.long())
			history.update(n=img_anchor.shape[0], name='val_acc@1', val=metric1.item())
			metric5 = metrics['acc@5'](y_pre, label_anchor.long(), topk=5)
			history.update(n=img_anchor.shape[0], name='val_acc@5', val=metric5.item())

			loss_classification = torch.mean(loss_classification)
			history.update(n=img_anchor.shape[0], name='val_Loss_Classification', val=loss_classification.item())
			# loss_compare = torch.mean(loss_compare)
			# history.update(n=img_anchor.shape[0], name='val_Loss_Distance', val=loss_compare.item())

		history.display(epoch=epoch+1, mode='validate')