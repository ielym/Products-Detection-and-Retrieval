import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align
import math

def focal_bce(predicts, targets, gamma=2, reduction='mean'):
	predicts = torch.sigmoid(predicts.view(-1))
	loss = - torch.pow(1-predicts, gamma) * targets *  torch.log(predicts+1e-8) - torch.pow(predicts, gamma) * (1-targets) * torch.log(1-predicts+1e-8)
	if reduction == 'mean':
		loss = torch.mean(loss)
	else:
		loss = torch.sum(loss)
	return loss


class FastPredictor(nn.Module):
	def __init__(self, num_classes, in_channels, hidden_channels):
		super(FastPredictor, self).__init__()

		self.max_pool = nn.AdaptiveAvgPool2d((1, 1))

		self.fc1 = nn.Linear(in_features=in_channels, out_features=hidden_channels)
		self.bn1 = nn.BatchNorm1d(num_features=hidden_channels)

		self.predictor_clss = nn.Linear(in_features=hidden_channels, out_features=num_classes)
		self.predictor_xyxy = nn.Linear(in_features=hidden_channels, out_features=num_classes * 4)
		self.__initial_weights()

	def __initial_weights(self):
		for m in self.children():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

	def forward(self, x):

		x = self.max_pool(x)
		x = torch.flatten(x, 1)
		x = F.relu(self.bn1(self.fc1(x)))

		predict_clss = self.predictor_clss(x)
		predict_dxdydwdh = self.predictor_xyxy(x)
		return predict_clss, predict_dxdydwdh

class FastHead(nn.Module):

	def __init__(self, args, tail, backbone_last_channels):
		super(FastHead, self).__init__()
		self.mode = args.mode

		self.fast_positive_iou_threshold = args.fast_positive_iou_threshold
		self.fast_negative_iou_threshold = args.fast_negative_iou_threshold
		self.fast_num_samples = args.fast_num_samples
		self.fast_pn_fraction = args.fast_pn_fraction
		self.num_classes = args.num_classes
		self.remove_min_score = args.remove_min_score

		self.roi_height = args.roi_height
		self.roi_width = args.roi_width

		# self.roipooling = ROIPooling(args.roi_height, args.roi_width)
		self.tail = tail
		self.predictor = FastPredictor(args.num_classes+1, backbone_last_channels, args.fast_hidden)

	def cal_batch_iou(self, xyxy1, xyxy2):
		flatten_xyxy1 = xyxy1.view(-1, 4) # torch.Size([9216, 4])
		flatten_xyxy2 = xyxy2.view(-1, 4) # torch.Size([25, 4])

		area1 = (flatten_xyxy1[..., 2] - flatten_xyxy1[..., 0]) * (flatten_xyxy1[..., 3] - flatten_xyxy1[..., 1]) # torch.Size([9216])
		area2 = (flatten_xyxy2[..., 2] - flatten_xyxy2[..., 0]) * (flatten_xyxy2[..., 3] - flatten_xyxy2[..., 1]) # torch.Size([25])

		x_min_c = torch.max(flatten_xyxy1[..., None, 0], flatten_xyxy2[..., 0]) # torch.Size([9216, 25])
		y_min_c = torch.max(flatten_xyxy1[..., None, 1], flatten_xyxy2[..., 1]) # torch.Size([9216, 25])
		x_max_c = torch.min(flatten_xyxy1[..., None, 2], flatten_xyxy2[..., 2]) # torch.Size([9216, 25])
		y_max_c = torch.min(flatten_xyxy1[..., None, 3], flatten_xyxy2[..., 3]) # torch.Size([9216, 25])

		inter = (x_max_c - x_min_c).clamp(min=0) * (y_max_c - y_min_c).clamp(min=0) # torch.Size([9216, 25])

		iou = inter / (area1[..., None] + area2 - inter) # torch.Size([9216, 25])

		return iou

	def cal_iou(self, xyxy1, xyxy2):
		flatten_xyxy1 = xyxy1.view(-1, 4)
		flatten_xyxy2 = xyxy2.view(-1, 4)

		area1 = (flatten_xyxy1[..., 2] - flatten_xyxy1[..., 0]) * (flatten_xyxy1[..., 3] - flatten_xyxy1[..., 1])
		area2 = (flatten_xyxy2[..., 2] - flatten_xyxy2[..., 0]) * (flatten_xyxy2[..., 3] - flatten_xyxy2[..., 1])

		x_min_c = torch.max(flatten_xyxy1[..., 0], flatten_xyxy2[..., 0])
		y_min_c = torch.max(flatten_xyxy1[..., 1], flatten_xyxy2[..., 1])
		x_max_c = torch.min(flatten_xyxy1[..., 2], flatten_xyxy2[..., 2])
		y_max_c = torch.min(flatten_xyxy1[..., 3], flatten_xyxy2[..., 3])

		inter = (x_max_c - x_min_c).clamp(min=0) * (y_max_c - y_min_c).clamp(min=0)

		iou = inter / (area1 + area2 - inter)

		return iou

	def get_positive_negative(self, proposal_batch_xyxy, target_batch_xyxy, target_batch_clss):
		'''
		:param proposal_batch_xyxy: torch.Size([314, 4])
		:param target_batch_xyxy: torch.Size([25, 4])
		:param target_batch_clss: torch.Size([25])
		:return:
		'''
		iou_between_proposal_and_target = self.cal_batch_iou(proposal_batch_xyxy.clone(), target_batch_xyxy.clone()) # torch.Size([314, 25])
		max_iou_values_for_target, matches = torch.max(iou_between_proposal_and_target, dim=1) # torch.Size([314])

		matches[max_iou_values_for_target <= self.fast_negative_iou_threshold] = -1
		matches[(max_iou_values_for_target > self.fast_negative_iou_threshold) & (max_iou_values_for_target < self.fast_positive_iou_threshold)] = -2

		positive_idxs = torch.nonzero(matches>=0).view(-1) # torch.Size([1, 1])
		negative_idxs = torch.nonzero(matches==-1).view(-1) # torch.Size([313, 1])
		positive_proposal_xyxy = proposal_batch_xyxy[positive_idxs, ...] # torch.Size([1, 4])
		negative_proposal_xyxy = proposal_batch_xyxy[negative_idxs, ...] # torch.Size([313, 4])
		positive_target_xyxy = target_batch_xyxy[matches[positive_idxs], ...] # torch.Size([1, 4])
		positive_target_clss = target_batch_clss[matches[positive_idxs], ...] # torch.Size([1])
		negative_target_clss = torch.zeros_like(negative_idxs).to(positive_target_clss) # torch.Size([313])

		return positive_proposal_xyxy, negative_proposal_xyxy, positive_target_xyxy, positive_target_clss, negative_target_clss

	def sample_positive_negative(self, proposals, targets, feature_height, feature_width):
		num_batch = len(proposals)

		positive_proposal_xyxy_batches = []
		negative_proposal_xyxy_batches = []
		positive_target_xyxy_batches = []
		positive_target_clss_batches = []
		negative_target_clss_batches = []
		positive_record_batches = []
		negative_record_batches = []
		for batch_idx in range(num_batch):
			proposal_batch_xyxy = proposals[batch_idx] # torch.Size([314, 4])
			target_batch = targets[batch_idx] # torch.Size([200, 5])
			target_batch = target_batch[target_batch[..., 4]>0] # torch.Size([25, 5])
			target_batch_xyxy = target_batch[..., :4] # torch.Size([25, 4])
			target_batch_clss = target_batch[..., 4] # torch.Size([25])
			target_batch_xyxy[..., 0] = target_batch_xyxy[..., 0] * feature_width
			target_batch_xyxy[..., 1] = target_batch_xyxy[..., 1] * feature_height
			target_batch_xyxy[..., 2] = target_batch_xyxy[..., 2] * feature_width
			target_batch_xyxy[..., 3] = target_batch_xyxy[..., 3] * feature_height

			positive_proposal_xyxy, negative_proposal_xyxy, positive_target_xyxy, positive_target_clss, negative_target_clss = \
				self.get_positive_negative(proposal_batch_xyxy.clone(), target_batch_xyxy.clone(), target_batch_clss.clone())
			positive_record = torch.zeros_like(positive_target_clss).fill_(batch_idx) # torch.Size([1])
			negative_record = torch.zeros_like(negative_target_clss).fill_(batch_idx) # torch.Size([313])

			positive_proposal_xyxy_batches.append(positive_proposal_xyxy) # torch.Size([1, 4])
			negative_proposal_xyxy_batches.append(negative_proposal_xyxy) # torch.Size([313, 4])
			positive_target_xyxy_batches.append(positive_target_xyxy) # torch.Size([1, 4])
			positive_target_clss_batches.append(positive_target_clss) # torch.Size([1])
			negative_target_clss_batches.append(negative_target_clss) # torch.Size([313])
			positive_record_batches.append(positive_record) # torch.Size([1])
			negative_record_batches.append(negative_record) # torch.Size([313])
		positive_proposal_xyxy_stack = torch.cat(positive_proposal_xyxy_batches, dim=0) # torch.Size([1, 4])
		negative_proposal_xyxy_stack = torch.cat(negative_proposal_xyxy_batches, dim=0) # torch.Size([1205, 4])
		positive_target_xyxy_stack = torch.cat(positive_target_xyxy_batches, dim=0) # torch.Size([1, 4])
		positive_target_clss_stack = torch.cat(positive_target_clss_batches, dim=0) # torch.Size([1])
		negative_target_clss_stack = torch.cat(negative_target_clss_batches, dim=0) # torch.Size([1205])
		positive_record_stack = torch.cat(positive_record_batches, dim=0) # torch.Size([1])
		negative_record_stack = torch.cat(negative_record_batches, dim=0) # torch.Size([1205])

		sampled_positive_nums = int(num_batch * self.fast_num_samples * self.fast_pn_fraction)
		sampled_positive_nums = min(positive_record_stack.size(0), sampled_positive_nums)
		sampled_negative_nums = int(num_batch * self.fast_num_samples - sampled_positive_nums)
		sampled_negative_nums = min(negative_record_stack.size(0), sampled_negative_nums)

		perm_positives = torch.randperm(positive_record_stack.size(0))[:sampled_positive_nums] # torch.Size([1])
		perm_negatives = torch.randperm(negative_record_stack.size(0))[:sampled_negative_nums] #torch.Size([255])

		positive_proposal_xyxy_stack = positive_proposal_xyxy_stack[perm_positives, ...] # torch.Size([1, 4])
		negative_proposal_xyxy_stack = negative_proposal_xyxy_stack[perm_negatives, ...] # torch.Size([255, 4])
		positive_target_xyxy_stack = positive_target_xyxy_stack[perm_positives, ...] # torch.Size([1, 4])
		positive_target_clss_stack = positive_target_clss_stack[perm_positives, ...] # torch.Size([1])
		negative_target_clss_stack = negative_target_clss_stack[perm_negatives, ...] # torch.Size([255])
		positive_record_stack = positive_record_stack[perm_positives, ...] # torch.Size([1])
		negative_record_stack = negative_record_stack[perm_negatives, ...] # torch.Size([255])

		pn_proposal_xyxy = torch.cat([positive_proposal_xyxy_stack, negative_proposal_xyxy_stack], dim=0) # torch.Size([256, 4])
		pn_target_clss = torch.cat([positive_target_clss_stack, negative_target_clss_stack], dim=0) # torch.Size([256])
		pn_record = torch.cat([positive_record_stack, negative_record_stack], dim=0) # torch.Size([256])

		return pn_proposal_xyxy, pn_target_clss, pn_record, positive_target_xyxy_stack

	def boxencoder(self, proposal_xyxy, target_xyxy):
		Px = (proposal_xyxy[..., 0] + proposal_xyxy[..., 2]) * 0.5
		Py = (proposal_xyxy[..., 1] + proposal_xyxy[..., 3]) * 0.5
		Pw = proposal_xyxy[..., 2] - proposal_xyxy[..., 0]
		Ph = proposal_xyxy[..., 3] - proposal_xyxy[..., 1]

		Gx = (target_xyxy[..., 0] + target_xyxy[..., 2]) * 0.5
		Gy = (target_xyxy[..., 1] + target_xyxy[..., 3]) * 0.5
		Gw = target_xyxy[..., 2] - target_xyxy[..., 0]
		Gh = target_xyxy[..., 3] - target_xyxy[..., 1]

		tx = (Gx - Px) / Pw # torch.Size([2])
		ty = (Gy - Py) / Ph
		tw = torch.log(Gw / Pw)
		th = torch.log(Gh / Ph)
		txtytwth = torch.cat([tx.unsqueeze(1), ty.unsqueeze(1), tw.unsqueeze(1), th.unsqueeze(1)], dim=1)
		return txtytwth

	def boxdecoder(self, dxdydwdh, xyxy):
		dx = dxdydwdh[..., 0]
		dy = dxdydwdh[..., 1]
		dw = dxdydwdh[..., 2]
		dh = dxdydwdh[..., 3]

		Px = (xyxy[..., 2] + xyxy[..., 0]) * 0.5
		Py = (xyxy[..., 3] + xyxy[..., 1]) * 0.5
		Pw = xyxy[..., 2] - xyxy[..., 0]
		Ph = xyxy[..., 3] - xyxy[..., 1]

		'''
		tx = (Gx - Px) / Pw
		tw = log(Gw / Pw)
		'''
		Gx = dx * Pw + Px
		Gy = dy * Ph + Py
		Gw = torch.exp(dw) * Pw
		Gh = torch.exp(dh) * Ph
		xywh = torch.cat([Gx.unsqueeze(1), Gy.unsqueeze(1), Gw.unsqueeze(1), Gh.unsqueeze(1)], dim=-1)
		return xywh

	def trans_xywh2xyxy(self, xywh):
		xy_min = xywh[..., :2] - xywh[..., 2:] * 0.5 # torch.Size([313, 2])
		xy_max = xywh[..., :2] + xywh[..., 2:] * 0.5
		xyxy = torch.cat([xy_min, xy_max], dim=1)
		return xyxy

	def IOU_Loss(self, target_xyxy, predict_xyxy, reduction='mean'):
		iou = self.cal_iou(target_xyxy, predict_xyxy)
		loss = 1 - iou
		if reduction == 'mean':
			loss = torch.mean(loss)
		else:
			loss = torch.sum(loss)
		return loss

	def GIOU_Loss(self, target_xyxy, predict_xyxy, reduction='mean'):
		area1 = (target_xyxy[..., 2] - target_xyxy[..., 0]) * (target_xyxy[..., 3] - target_xyxy[..., 1])
		area2 = (predict_xyxy[..., 2] - predict_xyxy[..., 0]) * (predict_xyxy[..., 3] - predict_xyxy[..., 1])

		x_min_i = torch.max(target_xyxy[..., 0], predict_xyxy[..., 0])
		y_min_i = torch.max(target_xyxy[..., 1], predict_xyxy[..., 1])
		x_max_i = torch.min(target_xyxy[..., 2], predict_xyxy[..., 2])
		y_max_i = torch.min(target_xyxy[..., 3], predict_xyxy[..., 3])
		inter = (x_max_i - x_min_i).clamp(min=0) * (y_max_i - y_min_i).clamp(min=0)

		iou = inter / (area1 + area2 - inter)

		x_min_c = torch.min(target_xyxy[..., 0], predict_xyxy[..., 0])
		y_min_c = torch.min(target_xyxy[..., 1], predict_xyxy[..., 1])
		x_max_c = torch.max(target_xyxy[..., 2], predict_xyxy[..., 2])
		y_max_c = torch.max(target_xyxy[..., 3], predict_xyxy[..., 3])
		C = (x_max_c - x_min_c).clamp(min=0) * (y_max_c - y_min_c).clamp(min=0)

		GIOU = iou - (C - (area1 + area2 - inter)) / C

		loss = 1 - GIOU

		if reduction == 'mean':
			loss = torch.mean(loss)
		else:
			loss = torch.sum(loss)
		return loss

	def DIOU_Loss(self, target_xyxy, predict_xyxy, reduction='mean'):
		area1 = (target_xyxy[..., 2] - target_xyxy[..., 0]) * (target_xyxy[..., 3] - target_xyxy[..., 1])
		area2 = (predict_xyxy[..., 2] - predict_xyxy[..., 0]) * (predict_xyxy[..., 3] - predict_xyxy[..., 1])

		x_min_i = torch.max(target_xyxy[..., 0], predict_xyxy[..., 0])
		y_min_i = torch.max(target_xyxy[..., 1], predict_xyxy[..., 1])
		x_max_i = torch.min(target_xyxy[..., 2], predict_xyxy[..., 2])
		y_max_i = torch.min(target_xyxy[..., 3], predict_xyxy[..., 3])
		inter = (x_max_i - x_min_i).clamp(min=0) * (y_max_i - y_min_i).clamp(min=0)

		iou = inter / (area1 + area2 - inter)

		x_min_c = torch.min(target_xyxy[..., 0], predict_xyxy[..., 0])
		y_min_c = torch.min(target_xyxy[..., 1], predict_xyxy[..., 1])
		x_max_c = torch.max(target_xyxy[..., 2], predict_xyxy[..., 2])
		y_max_c = torch.max(target_xyxy[..., 3], predict_xyxy[..., 3])
		c_2 = (x_max_c - x_min_c)**2 + (y_max_c - y_min_c)**2

		x_center_target = (target_xyxy[..., 0] + target_xyxy[..., 2]) * 0.5
		y_center_target = (target_xyxy[..., 1] + target_xyxy[..., 3]) * 0.5
		x_center_predict = (predict_xyxy[..., 0] + predict_xyxy[..., 2]) * 0.5
		y_center_predict = (predict_xyxy[..., 1] + predict_xyxy[..., 3]) * 0.5
		d_2 = (x_center_target - x_center_predict)**2 + (y_center_target - y_center_predict)**2

		loss = 1 - iou + d_2/c_2

		if reduction == 'mean':
			loss = torch.mean(loss)
		else:
			loss = torch.sum(loss)
		return loss

	def CIOU_Loss(self, target_xyxy, predict_xyxy, reduction='mean'):
		area1 = (target_xyxy[..., 2] - target_xyxy[..., 0]) * (target_xyxy[..., 3] - target_xyxy[..., 1])
		area2 = (predict_xyxy[..., 2] - predict_xyxy[..., 0]) * (predict_xyxy[..., 3] - predict_xyxy[..., 1])
		x_min_i = torch.max(target_xyxy[..., 0], predict_xyxy[..., 0])
		y_min_i = torch.max(target_xyxy[..., 1], predict_xyxy[..., 1])
		x_max_i = torch.min(target_xyxy[..., 2], predict_xyxy[..., 2])
		y_max_i = torch.min(target_xyxy[..., 3], predict_xyxy[..., 3])
		inter = (x_max_i - x_min_i).clamp(min=0) * (y_max_i - y_min_i).clamp(min=0)
		iou = inter / (area1 + area2 - inter)

		x_min_c = torch.min(target_xyxy[..., 0], predict_xyxy[..., 0])
		y_min_c = torch.min(target_xyxy[..., 1], predict_xyxy[..., 1])
		x_max_c = torch.max(target_xyxy[..., 2], predict_xyxy[..., 2])
		y_max_c = torch.max(target_xyxy[..., 3], predict_xyxy[..., 3])
		c_2 = (x_max_c - x_min_c)**2 + (y_max_c - y_min_c)**2
		x_center_target = (target_xyxy[..., 0] + target_xyxy[..., 2]) * 0.5
		y_center_target = (target_xyxy[..., 1] + target_xyxy[..., 3]) * 0.5
		x_center_predict = (predict_xyxy[..., 0] + predict_xyxy[..., 2]) * 0.5
		y_center_predict = (predict_xyxy[..., 1] + predict_xyxy[..., 3]) * 0.5
		d_2 = (x_center_target - x_center_predict)**2 + (y_center_target - y_center_predict)**2
		R = d_2 / c_2

		w_target = target_xyxy[..., 2] - target_xyxy[..., 0]
		h_target = target_xyxy[..., 3] - target_xyxy[..., 1]
		argtan_wh_target = torch.atan(w_target / h_target)
		w_predict = predict_xyxy[..., 2] - predict_xyxy[..., 0]
		h_predict = predict_xyxy[..., 3] - predict_xyxy[..., 1]
		argtan_wh_predict = torch.atan(w_predict / h_predict)
		v = (4 / math.pi**2) * (argtan_wh_target - argtan_wh_predict) ** 2
		alpha = v / (1 - iou + v).clamp(min=1e-8)

		ciou = iou - R - alpha * v
		ciou = torch.clamp(ciou, min=-1.0, max=1.0)
		loss = 1 - ciou

		if reduction == 'mean':
			loss = torch.mean(loss)
		else:
			loss = torch.sum(loss)
		return loss

	def ciou_loss(self, preds, bbox, eps=1e-7, reduction='mean'):
		'''
		https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
		:param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
		:param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
		:param eps: eps to avoid divide 0
		:param reduction: mean or sum
		:return: diou-loss
		'''
		ix1 = torch.max(preds[:, 0], bbox[:, 0])
		iy1 = torch.max(preds[:, 1], bbox[:, 1])
		ix2 = torch.min(preds[:, 2], bbox[:, 2])
		iy2 = torch.min(preds[:, 3], bbox[:, 3])

		iw = (ix2 - ix1 + 1.0).clamp(min=0.)
		ih = (iy2 - iy1 + 1.0).clamp(min=0.)

		# overlaps
		inters = iw * ih

		# union
		uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
					bbox[:, 2] - bbox[:, 0] + 1.0) * (
					  bbox[:, 3] - bbox[:, 1] + 1.0) - inters

		# iou
		iou = inters / (uni + eps)

		# inter_diag
		cxpreds = (preds[:, 2] + preds[:, 0]) / 2
		cypreds = (preds[:, 3] + preds[:, 1]) / 2

		cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
		cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

		inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

		# outer_diag
		ox1 = torch.min(preds[:, 0], bbox[:, 0])
		oy1 = torch.min(preds[:, 1], bbox[:, 1])
		ox2 = torch.max(preds[:, 2], bbox[:, 2])
		oy2 = torch.max(preds[:, 3], bbox[:, 3])

		outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

		diou = iou - inter_diag / outer_diag

		# calculate v,alpha
		wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
		hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
		wpreds = preds[:, 2] - preds[:, 0] + 1.0
		hpreds = preds[:, 3] - preds[:, 1] + 1.0
		v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
		alpha = v / (1 - iou + v)
		ciou = diou - alpha * v
		ciou = torch.clamp(ciou, min=-1.0, max=1.0)

		ciou_loss = 1 - ciou
		if reduction == 'mean':
			loss = torch.mean(ciou_loss)
		elif reduction == 'sum':
			loss = torch.sum(ciou_loss)
		else:
			raise NotImplementedError
		return loss

	def compute_loss(self, pn_proposal_xyxy, pn_target_clss, positive_target_xyxy_stack, predict_clss, predict_dxdydwdh):
		'''
		:param pn_proposal_xyxy: torch.Size([256, 4])
		:param pn_target_clss: torch.Size([256])
		:param pn_record: torch.Size([256])
		:param positive_target_xyxy_stack: torch.Size([2, 4])
		:param predict_clss: torch.Size([256, 117])
		:param predict_dxdydwdh: torch.Size([256, 117, 4])
		:return:
		'''
		# loss clss
		loss_clss = F.cross_entropy(predict_clss, pn_target_clss.long(), reduction='mean')
		# loss_clss = focal_bce(predict_clss, pn_target_clss, gamma=2, reduction='mean')

		# loss bbox
		if positive_target_xyxy_stack.size(0) == 0:
			return loss_clss * 0, loss_clss * 0

		positive_mask = (pn_target_clss > 0).float() # torch.Size([256])
		positive_target_clss = pn_target_clss[positive_mask==1, ...] # torch.Size([2])

		positive_proposal_xyxy = pn_proposal_xyxy[positive_mask==1, ...] # torch.Size([2, 4])
		positive_predict_dxdydwdh = predict_dxdydwdh[positive_mask==1, positive_target_clss.long(), ...] # torch.Size([2, 4])

		positive_predict_xywh = self.boxdecoder(positive_predict_dxdydwdh, positive_proposal_xyxy)
		positive_predict_xyxy = self.trans_xywh2xyxy(positive_predict_xywh)

		positive_target_xyxy = positive_target_xyxy_stack # torch.Size([2, 4])

		# loss_bbox = self.ciou_loss(positive_target_xyxy, positive_predict_xyxy, reduction='sum') / predict_clss.size(0)
		loss_bbox = self.ciou_loss(positive_predict_xyxy, positive_target_xyxy, reduction='sum') / predict_clss.size(0)

		return loss_clss, loss_bbox

	def post_process(self, predict_clss_batch, predict_dxdydwdh_batch, proposal_batch, input_size, feature_size):
		'''
		:param predict_clss_batch: torch.Size([313, 117])
		:param predict_dxdydwdh_batch: torch.Size([313, 117, 4])
		:param proposal_batch: torch.Size([313, 4])
		:return:
		'''
		input_height, input_width = input_size
		feature_height, feature_width = feature_size

		scores = torch.softmax(predict_clss_batch, dim=1) # torch.Size([313])
		scores, categories = torch.max(scores, dim=1) # torch.Size([313])

		positive_idxs = torch.nonzero(categories > 0).view(-1) # torch.Size([313])
		scores = scores[positive_idxs, ...] # torch.Size([313])
		categories = categories[positive_idxs, ...] # torch.Size([313])
		dxdydwdh = predict_dxdydwdh_batch[positive_idxs, categories, ...] # torch.Size([313, 4])
		proposal_xyxy = proposal_batch[positive_idxs, ...] # torch.Size([313, 4])

		xywh = self.boxdecoder(dxdydwdh, proposal_xyxy) # torch.Size([313, 4])
		xyxy = self.trans_xywh2xyxy(xywh) # torch.Size([313, 4])
		xyxy[..., 0] = xyxy[..., 0] / feature_width * input_width
		xyxy[..., 1] = xyxy[..., 1] / feature_height * input_height
		xyxy[..., 2] = xyxy[..., 2] / feature_width * input_width
		xyxy[..., 3] = xyxy[..., 3] / feature_height * input_height

		# remove min scores
		keep = torch.nonzero(scores > self.remove_min_score).view(-1)
		categories = categories[keep] # torch.Size([309])
		scores = scores[keep] # torch.Size([309])
		xyxy = xyxy[keep, :] # torch.Size([309, 4])

		return scores, categories, xyxy

	def forward(self, features, proposals, input_size, feature_size, targets):
		num_batch = features.size(0)
		feature_height, feature_width = feature_size

		losses = {}
		predicted = []
		if self.mode == 'Train':
			pn_proposal_xyxy, pn_target_clss, pn_record, positive_target_xyxy_stack = self.sample_positive_negative(proposals, targets, feature_height, feature_width)
			cat_proposals = torch.cat([pn_record.unsqueeze(1), pn_proposal_xyxy], dim=1)
			# roi_features = roi_pool(features, cat_features, output_size=(self.roi_height, self.roi_width), spatial_scale=1.0)
			roi_features = roi_align(features, cat_proposals, output_size=(self.roi_height, self.roi_width), spatial_scale=1.0)

			tail_features = self.tail(roi_features) # torch.Size([256, 2048, 4, 4])
			predict_clss, predict_dxdydwdh = self.predictor(tail_features) # torch.Size([256, 117]) torch.Size([256, 468])
			predict_dxdydwdh = predict_dxdydwdh.view(predict_dxdydwdh.size(0), -1, 4) # torch.Size([256, 117, 4])
			clss_loss, bbox_loss = self.compute_loss(pn_proposal_xyxy.clone(), pn_target_clss.clone(), positive_target_xyxy_stack.clone(), predict_clss.clone(), predict_dxdydwdh.clone())
			losses['clss_loss'] = clss_loss
			losses['bbox_loss'] = bbox_loss
		else:
			proposals = proposals[0]
			batch_record = torch.zeros([proposals.size(0), 1]).to(proposals)
			cat_proposals = torch.cat([batch_record, proposals], dim=1)
			roi_features = roi_align(features, cat_proposals, output_size=(self.roi_height, self.roi_width), spatial_scale=1.0)
			tail_features = self.tail(roi_features)  # torch.Size([313, 2048, 4, 4])
			predict_clss, predict_dxdydwdh = self.predictor(tail_features)  # torch.Size([313, 117]) torch.Size([313, 468])
			predict_dxdydwdh = predict_dxdydwdh.view(predict_dxdydwdh.size(0), -1, 4)  # torch.Size([313, 117, 4])
			scores, categories, xyxy = self.post_process(predict_clss, predict_dxdydwdh, proposals, input_size, feature_size)
			predicted = (scores, categories, xyxy)
		return predicted, losses