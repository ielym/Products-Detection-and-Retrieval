#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def weighted_mse(predicts, targets, weights=None, reduction='mean'):
	loss = 0.5 * torch.pow(predicts - targets, 2)
	loss = torch.sum(loss, dim=1)
	loss = loss * weights
	if reduction == 'mean':
		loss = torch.mean(loss)
	else:
		loss = torch.sum(loss)
	return loss

def weighted_bce(predicts, targets, weights=None, reduction='mean'):
	loss = -targets * torch.log(predicts+1e-8) - (1-targets) * torch.log(1-predicts+1e-8)
	loss = torch.sum(loss, dim=1)
	loss = loss * weights
	if reduction == 'mean':
		loss = torch.mean(loss)
	else:
		loss = torch.sum(loss)
	return loss

def focal_bce(predicts, targets, alpha=0.2, gamma=2, reduction='mean'):
	loss = - (1-alpha) * torch.pow(1-predicts, gamma) * targets *  torch.log(predicts+1e-8) - alpha * torch.pow(predicts, gamma) * (1-targets) * torch.log(1-predicts+1e-8)
	if reduction == 'mean':
		loss = torch.mean(loss)
	else:
		loss = torch.sum(loss)
	return loss


def get_offsets(feature_size_levels, num_anchor_levels, device):
	num_levels = len(feature_size_levels)

	offset_levels = []
	for level_idx in range(num_levels):
		feature_size = feature_size_levels[level_idx]
		num_anchor = num_anchor_levels[level_idx]

		offset_x = torch.arange(0, feature_size).float()
		offset_y = torch.arange(0, feature_size).float()
		offset_y, offset_x  = torch.meshgrid([offset_x, offset_y])
		offsets = torch.cat([offset_x.unsqueeze(2), offset_y.unsqueeze(2)], dim=2).unsqueeze(2).to(device) # torch.Size([52, 52, 1, 2])
		offsets = offsets.expand([feature_size, feature_size, num_anchor, 2]) # torch.Size([52, 52, 3, 2])
		offset_levels.append(offsets)
	return offset_levels

def boxdecoder(dxdy, dwdh, anchors):
	wh = torch.exp(dwdh) * anchors # torch.Size([52, 52, 3, 2])
	xywh = torch.cat([dxdy, wh], dim=3) # torch.Size([52, 52, 3, 4])
	return xywh

def transfor_xywh2xyxy(xywh, offset_xy):
	xy_center = xywh[..., :2] + offset_xy
	wh = xywh[..., 2:].clone()

	xy_min = xy_center - wh * 0.5 # torch.Size([52, 52, 3, 2])
	xy_max = xy_center + wh * 0.5 # torch.Size([52, 52, 3, 2])

	xyxy = torch.cat([xy_min, xy_max], dim=3)
	return xyxy

def transfor_xyxy2xywh(xyxy):
	xy_min = xyxy[..., :2] # torch.Size([5, 2])
	xy_max = xyxy[..., 2:] # torch.Size([5, 2])

	xy_center = (xy_max + xy_min) * 0.5 # torch.Size([5, 2])
	xy_ij = torch.floor(xy_center)

	xy_offset = xy_center - xy_ij # torch.Size([5, 2])
	wh = xy_max - xy_min # torch.Size([5, 2])

	xywh = torch.cat([xy_offset, wh], dim=1)
	return xywh, xy_ij

def cal_batch_iou(xyxy1, xyxy2):
	'''
	:param xyxy1: torch.Size([52, 52, 3, 4])
	:param xyxy2: torch.Size([5, 4])
	:return:
	'''
	flatten_xyxy1 = xyxy1.view(-1, 4) # torch.Size([8112, 4])
	flatten_xyxy2 = xyxy2.view(-1, 4) # torch.Size([5, 4])

	area1 = (flatten_xyxy1[:, 2] - flatten_xyxy1[:, 0]) * (flatten_xyxy1[:, 3] - flatten_xyxy1[:, 1]) # torch.Size([8112])
	area2 = (flatten_xyxy2[:, 2] - flatten_xyxy2[:, 0]) * (flatten_xyxy2[:, 3] - flatten_xyxy2[:, 1]) # torch.Size([5])

	x_min = torch.max(flatten_xyxy1[:, None, 0], flatten_xyxy2[:, 0]) # torch.Size([8112, 5])
	y_min = torch.max(flatten_xyxy1[:, None, 1], flatten_xyxy2[:, 1]) # torch.Size([8112, 5])
	x_max = torch.min(flatten_xyxy1[:, None, 2], flatten_xyxy2[:, 2]) # torch.Size([8112, 5])
	y_max = torch.min(flatten_xyxy1[:, None, 3], flatten_xyxy2[:, 3]) # torch.Size([8112, 5])

	inter = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0) # torch.Size([8112, 5])

	iou = inter / (area1[:, None] + area2 - inter) # torch.Size([8112, 5])
	return iou

def compute_batch_loss(target_batch, predict_batch, anchor_levels, offsets_levels, num_anchor_levels, feature_size_levels, bg_threshold, num_classes, iteration):
	num_levels = len(feature_size_levels)
	num_target = target_batch.size(0)

	target_xyxy_normal = target_batch[..., :4] # torch.Size([5, 4])
	target_clss = target_batch[..., 4] # torch.Size([5])

	# iou, which_level, i, j, which_anchor, clss, gt_xyxy, gt_xywh, anchor_wh
	response_iou_record = torch.zeros([num_target, 16], requires_grad=False).to(device=target_batch.device)

	iou_between_predict_and_target_levels = []
	mask_np_levels = []
	predict_conf_levels = []
	predict_clss_levels = []
	predict_dxdy_levels = []
	predict_dwdh_levels = []
	for level_idx in range(num_levels):
		num_anchor = num_anchor_levels[level_idx]
		feature_size = feature_size_levels[level_idx]

		offset_xy = offsets_levels[level_idx] # torch.Size([52, 52, 3, 2])

		anchor_wh = anchor_levels[level_idx] * feature_size # torch.Size([3, 2])

		target_xyxy = target_xyxy_normal * feature_size # torch.Size([5, 4])
		target_xywh, target_ij = transfor_xyxy2xywh(target_xyxy) # torch.Size([5, 4]) # torch.Size([5, 2])

		predict = predict_batch[level_idx] # torch.Size([75, 52, 52])
		predict = predict.permute(1, 2, 0) # torch.Size([52, 52, 75])

		predict = predict.view(feature_size, feature_size, num_anchor, 5+num_classes) # torch.Size([52, 52, 3, 95])

		predict_dxdy = torch.sigmoid(predict[..., 0:2]) # torch.Size([52, 52, 3, 2])
		predict_dwdh = predict[..., 2:4] # torch.Size([52, 52, 3, 2])
		predict_conf = torch.sigmoid(predict[..., 4]) # torch.Size([52, 52, 3])
		# predict_clss = torch.sigmoid(predict[..., 5:]) # torch.Size([52, 52, 3, 90])
		predict_clss = torch.softmax(predict[..., 5:], dim=3) # torch.Size([52, 52, 3, 90])

		predict_xywh = boxdecoder(predict_dxdy, predict_dwdh, anchor_wh) # torch.Size([52, 52, 3, 4])
		predict_xyxy = transfor_xywh2xyxy(predict_xywh, offset_xy) # torch.Size([52, 52, 3, 4])

		iou_between_predict_and_target = cal_batch_iou(predict_xyxy, target_xyxy) # torch.Size([8112, 2])
		iou_between_predict_and_target = iou_between_predict_and_target.view(feature_size, feature_size, num_anchor, num_target) # torch.Size([52, 52, 3, 2])
		iou_between_predict_and_target_levels.append(iou_between_predict_and_target)

		max_iou_values_for_each_anchor, _ = torch.max(iou_between_predict_and_target, dim=3) # torch.Size([52, 52, 3])
		mask_np = (max_iou_values_for_each_anchor < bg_threshold).float() # torch.Size([52, 52, 3])
		mask_np_levels.append(mask_np)

		predict_conf_levels.append(predict_conf.contiguous().view(-1, 1))
		predict_clss_levels.append(predict_clss.contiguous().view(-1, num_classes))
		predict_dxdy_levels.append(predict_dxdy.contiguous().view(-1, 2))
		predict_dwdh_levels.append(predict_dwdh.contiguous().view(-1, 2))

		for gt_idx in range(num_target):
			i, j = target_ij[gt_idx, ...].long()
			gt_xyxy = target_xyxy[gt_idx, ...]
			gt_xywh = target_xywh[gt_idx, ...]
			grid_ious = iou_between_predict_and_target[j, i, :, gt_idx] # torch.Size([3])
			max_iou, which_anchor = torch.max(grid_ious, dim=0)
			# iou, which_level, i, j, which_anchor, clss, gt_xyxy, gt_xywh, anchor_wh
			if max_iou >= response_iou_record[gt_idx, 0]:
				response_iou_record[gt_idx, 0] = max_iou
				response_iou_record[gt_idx, 1] = level_idx
				response_iou_record[gt_idx, 2] = i
				response_iou_record[gt_idx, 3] = j
				response_iou_record[gt_idx, 4] = which_anchor
				response_iou_record[gt_idx, 5] = target_clss[gt_idx]
				response_iou_record[gt_idx, 6:10] = gt_xyxy
				response_iou_record[gt_idx, 10:14] = gt_xywh
				response_iou_record[gt_idx, 14:] = anchor_wh[which_anchor, ...]

	#===========================================================  mask  ================================================
	# iou, which_level, i, j, which_anchor, clss, gt_xyxy, gt_xywh, anchor_wh
	for gt_idx in range(num_target):
		in_which_level = int(response_iou_record[gt_idx, 1])
		i = int(response_iou_record[gt_idx, 2])
		j = int(response_iou_record[gt_idx, 3])
		in_which_anchor = int(response_iou_record[gt_idx, 4])
		mask_np_levels[in_which_level][j, i, in_which_anchor] = gt_idx + 2
	mask_np_stack = torch.cat([i.contiguous().view(-1, 1) for i in mask_np_levels]).view(-1) # torch.Size([10647])
	gt_idxs = mask_np_stack[mask_np_stack >= 2].long() - 2

	# ================================================== elements ==================================================
	predict_conf_stack = torch.cat(predict_conf_levels, dim=0) # torch.Size([10647, 1])
	predict_clss_stack = torch.cat(predict_clss_levels, dim=0) # torch.Size([10647, 20])
	predict_dxdy_stack = torch.cat(predict_dxdy_levels, dim=0) # torch.Size([10647, 2])
	predict_dwdh_stack = torch.cat(predict_dwdh_levels, dim=0) # torch.Size([10647, 2])

	# ================================================== losses ==================================================
	# loss negative confidence
	predict_negative_conf = predict_conf_stack[mask_np_stack==1, ...] # torch.Size([10628, 1])
	target_negative_conf = torch.zeros_like(predict_negative_conf, requires_grad=False)
	loss_negative_conf = F.binary_cross_entropy(predict_negative_conf, target_negative_conf, reduction='sum')
	# loss_negative_conf = focal_bce(predict_negative_conf, target_negative_conf, reduction='sum')

	# loss response confidence
	predict_response_conf = predict_conf_stack[mask_np_stack>=2, ...] # torch.Size([2, 1])
	target_response_conf = torch.ones_like(predict_response_conf, requires_grad=False)
	loss_response_conf = F.binary_cross_entropy(predict_response_conf, target_response_conf, reduction='sum')
	# loss_response_conf = focal_bce(predict_response_conf, target_response_conf, reduction='sum')

	# loss response classes
	predict_response_clss = predict_clss_stack[mask_np_stack>=2, ...] # torch.Size([2, 20])
	target_response_clss_num = target_clss[gt_idxs, ...] - 1
	# target_response_clss = predict_response_clss.data.clone().zero_().scatter_(1, target_response_clss_num.view(-1, 1).long(), 1) # torch.Size([2, 20])
	# loss_response_clss = F.binary_cross_entropy(predict_response_clss, target_response_clss, reduction='sum')
	loss_response_clss = F.cross_entropy(predict_response_clss, target_response_clss_num.long(), reduction='sum')

	# loss response bboxes
	predict_response_dxdy = predict_dxdy_stack[mask_np_stack>=2, ...] # torch.Size([2, 2])
	predict_response_dwdh = predict_dwdh_stack[mask_np_stack>=2, ...] # torch.Size([2, 2])
	target_response_txty = response_iou_record[gt_idxs, 10:12] # torch.Size([2, 2])
	target_response_wh = response_iou_record[gt_idxs, 12:14] # torch.Size([2, 2])
	anchor_response_wh = response_iou_record[gt_idxs, 14:] # torch.Size([2, 2])
	target_response_twth = torch.log(target_response_wh / anchor_response_wh)

	loss_response_dxdy = F.binary_cross_entropy(predict_response_dxdy, target_response_txty.detach(), reduction='sum')
	loss_response_dwdy = F.smooth_l1_loss(predict_response_dwdh, target_response_twth, reduction='sum')

	# print(0.2*loss_negative_conf.item(), loss_response_conf.item(), 10*loss_response_clss.item(), 5*loss_response_dxdy.item(), 5*loss_response_dwdy.item())
	loss_batch = 0.2*loss_negative_conf + loss_response_conf + loss_response_clss + 5*(loss_response_dxdy + loss_response_dwdy)
	# loss_batch = 0*loss_negative_conf + 0*loss_response_conf + loss_response_clss + 0*(loss_response_dxdy + loss_response_dwdy)

	return loss_batch



def Loss_Yolo(predict_levels, targets, anchor_levels, num_anchor_levels, feature_size_levels, bg_threshold, num_classes, device, iteration):
	BS = targets.size(0) # torch.Size([4, 200, 5])
	num_levels = len(predict_levels)

	offsets_levels = get_offsets(feature_size_levels, num_anchor_levels, device)

	loss = 0
	for batch_idx in range(BS):
		target_batch = targets[batch_idx, ...] # torch.Size([200, 5])
		target_batch = target_batch[target_batch[..., 4]>0] # torch.Size([5, 5])
		if target_batch.size(0) == 0:
			BS -= 1
			continue
		predict_batch = [predict_levels[i][batch_idx, ...] for i in range(num_levels)]

		loss += compute_batch_loss(target_batch, predict_batch, anchor_levels, offsets_levels, num_anchor_levels, feature_size_levels, bg_threshold, num_classes, iteration)
	return loss / BS

def post_process(predict_levels, anchor_levels, num_anchor_levels, feature_size_levels, input_size, num_classes, device, prob_threshold):
	num_levels = len(predict_levels)

	offsets_levles = get_offsets(feature_size_levels, num_anchor_levels, device)  # torch.Size([676, 5, 2])

	prob_levels = []
	category_levels = []
	predict_xyxy_levels = []
	for level_idx in range(num_levels):
		num_anchors = num_anchor_levels[level_idx]
		feature_size = feature_size_levels[level_idx]

		anchors = anchor_levels[level_idx] * feature_size # torch.Size([3, 2])
		anchors = anchors.unsqueeze(0).unsqueeze(0)

		predict = predict_levels[level_idx].squeeze(0) # torch.Size([75, 52, 52])
		predict = predict.permute(1, 2, 0)  # torch.Size([52, 52, 75])
		predict = predict.view(feature_size, feature_size, num_anchors, 5+num_classes)  # torch.Size([52, 52, 3, 25])

		offset = offsets_levles[level_idx] # torch.Size([52, 52, 3, 2])


		# ====================================================decoding======================================================
		predict_dxdy = torch.sigmoid(predict[..., :2]) # torch.Size([52, 52, 3, 2])
		predict_dwdh = predict[..., 2:4] # torch.Size([52, 52, 3, 2])
		predict_conf = torch.sigmoid(predict[..., 4:5]) # torch.Size([52, 52, 3, 1])
		predict_clss  = torch.softmax(predict[..., 5:], dim=3) # torch.Size([52, 52, 3, 20])

		predict_xy = predict_dxdy + offset
		predict_wh = torch.exp(predict_dwdh) * anchors
		predict_min_xy = predict_xy - predict_wh * 0.5
		predict_max_xy = predict_xy + predict_wh * 0.5
		predict_xyxy = torch.cat([predict_min_xy, predict_max_xy], dim=3) # torch.Size([52, 52, 3, 4])
		predict_xyxy = predict_xyxy / feature_size * input_size

		confidence_score = predict_conf * predict_clss # torch.Size([52, 52, 3, 20])
		probs, categories = torch.max(confidence_score, dim=3) # torch.Size([52, 52, 3]) torch.Size([52, 52, 3])
		# ====================================================filters======================================================
		# remove min prob
		keep = (probs>=prob_threshold).float() # torch.Size([52, 52, 3])
		probs = probs[keep==1, ...] # torch.Size([37])
		categories = categories[keep==1, ...] # torch.Size([37])
		predict_xyxy = predict_xyxy[keep==1, ...] # torch.Size([37, 4])

		prob_levels.append(probs)
		category_levels.append(categories)
		predict_xyxy_levels.append(predict_xyxy)

	prob_levels = torch.cat(prob_levels, dim=0)
	category_levels = torch.cat(category_levels, dim=0)
	predict_xyxy_levels = torch.cat(predict_xyxy_levels, dim=0)
	results = (prob_levels, category_levels, predict_xyxy_levels)
	return results