#coding:utf-8
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class RPN_HEAD(nn.Module):
	def __init__(self, in_channels, num_anchor):
		super(RPN_HEAD, self).__init__()

		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True)

		self.bbox_predictor = nn.Conv2d(in_channels=in_channels, out_channels=num_anchor*4, kernel_size=1, stride=1, padding=0, bias=True)
		self.clss_predictor = nn.Conv2d(in_channels=in_channels, out_channels=num_anchor*1, kernel_size=1, stride=1, padding=0, bias=True)

		self._initial_layers()

	def _initial_layers(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		out = torch.relu(self.conv(x))

		predict_xywh = self.bbox_predictor(out)
		predict_clss = self.clss_predictor(out)

		return predict_xywh, predict_clss

class RegionProposalNetwork(nn.Module):
	def __init__(self, backbone_channels, args):
		super(RegionProposalNetwork, self).__init__()
		self.mode = args.mode
		self.num_anchor = args.num_anchor

		self.positive_threshold = args.rpn_positive_iou_threshold
		self.negative_threshold = args.rpn_negative_iou_threshold
		self.rpn_num_samples = args.rpn_num_samples
		self.rpn_pn_fraction = args.rpn_pn_fraction
		self.rpn_remove_min_size = args.rpn_remove_min_size
		self.rpn_pre_nms_top_n_train = args.rpn_pre_nms_top_n_train
		self.rpn_post_nms_top_n_train = args.rpn_post_nms_top_n_train
		self.rpn_pre_nms_top_n_test = args.rpn_pre_nms_top_n_test
		self.rpn_post_nms_top_n_test = args.rpn_post_nms_top_n_test
		self.rpn_nms_thresh = args.rpn_nms_thresh

		self.rpn_head = RPN_HEAD(in_channels=backbone_channels, num_anchor=args.num_anchor)

	def get_offsets(self, feature_height, feature_width, num_anchor, device):
		offset_x = torch.arange(0, feature_width).float()
		offset_y = torch.arange(0, feature_height).float()
		offset_y, offset_x = torch.meshgrid([offset_y, offset_x])
		offsets = torch.cat([offset_x.unsqueeze(2), offset_y.unsqueeze(2)], dim=2).float().unsqueeze(2).to(device) # torch.Size([32, 32, 1, 2])
		offsets = offsets.expand([feature_height, feature_width, num_anchor, 2]) # torch.Size([32, 32, 9, 2])
		return offsets

	def boxdecoder(self, dxdydwdh, pxpypwph):
		PxPy = pxpypwph[..., :2]
		PwPh = pxpypwph[..., 2:]
		dxdy = dxdydwdh[..., :2]  # torch.Size([4, 32, 32, 9, 2])
		dwdh = dxdydwdh[..., 2:]  # torch.Size([4, 32, 32, 9, 2])

		xy = dxdy * PwPh + PxPy  # torch.Size([4, 32, 32, 9, 2])
		wh = torch.exp(dwdh) * PwPh  # torch.Size([4, 32, 32, 9, 2])
		xywh = torch.cat([xy, wh], dim=4)  # torch.Size([4, 32, 32, 9, 4])
		return xywh

		xywh = torch.cat([Gx.unsqueeze(1), Gy.unsqueeze(1), Gw.unsqueeze(1), Gh.unsqueeze(1)], dim=1)
		return xywh

	def boxencoder(self, target_xyxy, anchor_xyxy):
		flatten_target_xyxy = target_xyxy.clone().view(-1, 4)  # torch.Size([9216, 4])
		flatten_anchor_xyxy = anchor_xyxy.clone().view(-1, 4)  # torch.Size([9216, 4])

		Gx = (flatten_target_xyxy[..., 2] + flatten_target_xyxy[..., 0]) * 0.5
		Gy = (flatten_target_xyxy[..., 3] + flatten_target_xyxy[..., 1]) * 0.5
		Gw = flatten_target_xyxy[..., 2] - flatten_target_xyxy[..., 0]
		Gh = flatten_target_xyxy[..., 3] - flatten_target_xyxy[..., 1]

		Px = (flatten_anchor_xyxy[..., 2] + flatten_anchor_xyxy[..., 0]) * 0.5
		Py = (flatten_anchor_xyxy[..., 3] + flatten_anchor_xyxy[..., 1]) * 0.5
		Pw = flatten_anchor_xyxy[..., 2] - flatten_anchor_xyxy[..., 0]
		Ph = flatten_anchor_xyxy[..., 3] - flatten_anchor_xyxy[..., 1]

		tx = (Gx - Px) / Pw
		ty = (Gy - Py) / Ph
		tw = torch.log(Gw / Pw)
		th = torch.log(Gh / Ph)

		txtytwth = torch.cat([tx.unsqueeze(1), ty.unsqueeze(1), tw.unsqueeze(1), th.unsqueeze(1)], dim=1)  # torch.Size([9216, 4])
		return txtytwth

	def trans_xywh2xyxy(self, xywh):
		xy_min = xywh[..., :2] - xywh[..., 2:] * 0.5
		xy_max = xywh[..., :2] + xywh[..., 2:] * 0.5
		xyxy = torch.cat([xy_min, xy_max], dim=-1)
		return xyxy

	def trans_xyxy2xywh(self, xyxy):
		xy = (xyxy[..., :2] + xyxy[..., 2:]) * 0.5
		wh = xyxy[..., 2:] - xyxy[..., :2]
		xywh = torch.cat([xy, wh], dim=1)
		return xywh

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

	def select_train_samples(self, predict_dxdydwdh_batch, predict_clss_batch, target_xyxy_batch, anchors_xyxy):
		'''
		:param predict_dxdydwdh_batch: torch.Size([32, 32, 9, 4])
		:param predict_clss_batch: torch.Size([32, 32, 9, 1])
		:param target_xyxy_batch: torch.Size([25, 4])
		:param anchors_xyxy: torch.Size([32, 32, 9, 4])
		:return:
		'''
		device = predict_dxdydwdh_batch.device

		predict_dxdydwdh = predict_dxdydwdh_batch.reshape(-1, 4) # torch.Size([9216, 4])
		predict_clss = predict_clss_batch.reshape(-1, 1) # torch.Size([9216, 1])
		target_xyxy = target_xyxy_batch.view(-1, 4) # torch.Size([25, 4])

		iou_between_anchor_and_target = self.cal_batch_iou(anchors_xyxy, target_xyxy) # torch.Size([9216, 25])

		# case 2
		max_iou_values_for_each_anchor, matches = torch.max(iou_between_anchor_and_target, dim=1) # torch.Size([9216])
		matches[max_iou_values_for_each_anchor<self.negative_threshold] = -1
		matches[(max_iou_values_for_each_anchor>=self.negative_threshold) & (max_iou_values_for_each_anchor<self.positive_threshold)] = -2

		# case 1
		_, max_iou_idxs_for_each_target = torch.max(iou_between_anchor_and_target, dim=0) # torch.Size([25])
		for target_idx in range(target_xyxy.size(0)):
			matches[max_iou_idxs_for_each_target[target_idx]] = target_idx

		target_xyxy = target_xyxy[matches.clamp(min=0), ...] # torch.Size([9216, 4])
		target_txtytwth = self.boxencoder(target_xyxy, anchors_xyxy) # torch.Size([9216, 4])

		target_clss = matches.clone() # torch.Size([9216])
		target_clss[matches>=0] = 1
		target_clss[matches==-1] = 0
		target_clss[matches==-2] = -1

		positive_idxs = torch.nonzero(target_clss==1).view(-1) # torch.Size([25])
		negative_idxs = torch.nonzero(target_clss==0).view(-1) # torch.Size([9191])

		num_positive = int(self.rpn_num_samples * self.rpn_pn_fraction)
		num_positive = min(num_positive, positive_idxs.size(0))
		num_negative = self.rpn_num_samples - num_positive
		num_negative = min(num_negative, negative_idxs.size(0))

		positive_perm = torch.randperm(positive_idxs.numel(), device=device)[:num_positive]
		negative_perm = torch.randperm(negative_idxs.numel(), device=device)[:num_negative]

		sampled_positive_idxs = positive_idxs[positive_perm]
		sampled_negative_idxs = negative_idxs[negative_perm]
		sampled_all_idxs = torch.cat([sampled_positive_idxs, sampled_negative_idxs], dim=0).view(-1) # torch.Size([256])

		sampled_positive_dxdydwdh = predict_dxdydwdh[sampled_positive_idxs, ...] # torch.Size([25, 4])
		sampled_positive_txtytwth = target_txtytwth[sampled_positive_idxs, ...] # torch.Size([25, 4])

		sampled_predict_clss = predict_clss[sampled_all_idxs, ...].view(-1, 1) # torch.Size([256, 1])
		sampled_target_clss = target_clss[sampled_all_idxs, ...].view(-1, 1) # torch.Size([256])
		return sampled_positive_dxdydwdh, sampled_positive_txtytwth, sampled_predict_clss, sampled_target_clss

	def compute_loss(self, dxdydwdh, txtytwth, predict_clss, target_clss):
		loss_clss = F.binary_cross_entropy_with_logits(predict_clss.float(), target_clss.float(), reduction='mean')
		loss_xywh = F.smooth_l1_loss(dxdydwdh, txtytwth, reduction='sum') / predict_clss.size(0)
		return loss_clss, loss_xywh

	def post_process(self, proposals_xyxy, predict_clss, feature_size):
		feature_height, feature_width = feature_size
		num_batches = predict_clss.size(0)

		final_proposals = []
		for batch_idx in range(num_batches):

			proposals_xyxy_batch = proposals_xyxy[batch_idx, ...]  # torch.Size([32, 32, 9, 4])
			predict_clss_batch = predict_clss[batch_idx, ...]  # torch.Size([32, 32, 9, 1])

			# limit height, width
			proposals_xyxy_batch[..., 0] = torch.clamp(proposals_xyxy_batch[..., 0].clone(), min=0, max=feature_width - 1)
			proposals_xyxy_batch[..., 1] = torch.clamp(proposals_xyxy_batch[..., 1].clone(), min=0, max=feature_height - 1)
			proposals_xyxy_batch[..., 2] = torch.clamp(proposals_xyxy_batch[..., 2].clone(), min=0, max=feature_width - 1)
			proposals_xyxy_batch[..., 3] = torch.clamp(proposals_xyxy_batch[..., 3].clone(), min=0, max=feature_height - 1)

			# remove min size
			ws = proposals_xyxy_batch[..., 2] - proposals_xyxy_batch[..., 0]  # torch.Size([36864])
			hs = proposals_xyxy_batch[..., 3] - proposals_xyxy_batch[..., 1]  # torch.Size([36864])
			keep = ((ws >= self.rpn_remove_min_size) & (hs >= self.rpn_remove_min_size)).float()
			proposals_xyxy_batch_keep = proposals_xyxy_batch[keep == 1, ...]  # torch.Size([7322, 4])
			predict_clss_keep = predict_clss_batch[keep == 1, ...]  # torch.Size([7322, 1])

			# pre_nms
			if self.mode == 'Train':
				pre_nms_top_n = self.rpn_pre_nms_top_n_train
			else:
				pre_nms_top_n = self.rpn_pre_nms_top_n_test
			pre_nms_top_n = min(pre_nms_top_n, proposals_xyxy_batch_keep.size(0))
			_, keep = predict_clss_keep.topk(k=pre_nms_top_n, dim=0)  # torch.Size([2000, 1])
			keep = keep.view(-1)
			proposals_xyxy_batch_keep = proposals_xyxy_batch_keep[keep, ...]  # torch.Size([2000, 4])
			predict_clss_keep = predict_clss_keep[keep, ...]  # torch.Size([2000, 1])

			# nms
			keep = torchvision.ops.nms(boxes=proposals_xyxy_batch_keep, scores=predict_clss_keep.view(-1), iou_threshold=self.rpn_nms_thresh)  # torch.Size([322])
			proposals_xyxy_batch_keep = proposals_xyxy_batch_keep[keep, ...]  # torch.Size([322, 4])
			predict_clss_keep = predict_clss_keep[keep, ...]  # torch.Size([322, 1])

			# post_nms
			if self.mode == 'Train':
				post_nms_top_n = self.rpn_post_nms_top_n_train
			else:
				post_nms_top_n = self.rpn_post_nms_top_n_test
			post_nms_top_n = min(post_nms_top_n, proposals_xyxy_batch_keep.size(0))
			_, keep = predict_clss_keep.topk(k=post_nms_top_n, dim=0)
			keep = keep.view(-1)
			proposals_xyxy_batch_keep = proposals_xyxy_batch_keep[keep, ...] # torch.Size([322, 4])

			final_proposals.append(proposals_xyxy_batch_keep)
		return final_proposals

	def forward(self, x, anchor_wh, input_size, feature_size, targets=None):
		device = x.device
		num_batches = x.size(0)
		feature_height, feature_width = feature_size

		predict_dxdydwdh, predict_clss = self.rpn_head(x) # torch.Size([4, 36, 32, 32]) torch.Size([4, 9, 32, 32])

		predict_dxdydwdh = predict_dxdydwdh.permute([0, 2, 3, 1]) # torch.Size([4, 32, 32, 36])
		predict_clss = predict_clss.permute([0, 2, 3, 1]) # torch.Size([4, 32, 32, 9])
		predict_dxdydwdh = predict_dxdydwdh.view(num_batches, feature_height, feature_width, self.num_anchor, 4)  # torch.Size([4, 32, 32, 9, 4])
		predict_clss = predict_clss.view(num_batches, feature_height, feature_width, self.num_anchor, 1)  # torch.Size([4, 32, 32, 9, 1])

		anchors = anchor_wh.repeat([feature_height, feature_width, 1, 1]).float().to(device=device) # torch.Size([32, 32, 9, 2])
		anchors[..., 0] = anchors[..., 0] * feature_width
		anchors[..., 1] = anchors[..., 1] * feature_height
		offsets = self.get_offsets(feature_height, feature_width, self.num_anchor, device) # torch.Size([32, 32, 9, 2])
		anchor_xywh = torch.cat([offsets, anchors], dim=3) # torch.Size([32, 32, 9, 4])
		anchors_xyxy = self.trans_xywh2xyxy(anchor_xywh) # torch.Size([32, 32, 9, 4])

		proposals_xywh = self.boxdecoder(predict_dxdydwdh, anchor_xywh) # torch.Size([4, 32, 32, 9, 4])
		proposals_xyxy = self.trans_xywh2xyxy(proposals_xywh) # torch.Size([4, 32, 32, 9, 4])
		finall_proposals_xyxy = self.post_process(proposals_xyxy, predict_clss, feature_size)

		losses = {}
		if self.mode == 'Train':
			sampled_positive_dxdydwdh = []
			sampled_positive_txtytwth = []
			sampled_predict_clss = []
			sampled_target_clss = []
			for batch_idx in range(num_batches):
				target_batch = targets[batch_idx, ...]
				target_batch = target_batch[target_batch[..., 4] > 0]
				target_xyxy_batch = target_batch[..., :4] # torch.Size([200, 4])
				target_xyxy_batch[..., 0] = target_xyxy_batch[..., 0] * feature_width
				target_xyxy_batch[..., 1] = target_xyxy_batch[..., 1] * feature_height
				target_xyxy_batch[..., 2] = target_xyxy_batch[..., 2] * feature_width
				target_xyxy_batch[..., 3] = target_xyxy_batch[..., 3] * feature_height

				predict_dxdydwdh_batch = predict_dxdydwdh[batch_idx, ...] # torch.Size([32, 32, 9, 4])
				predict_clss_batch = predict_clss[batch_idx, ...] # torch.Size([32, 32, 9, 1])
				choised_positive_dxdydwdh, choised_positive_txtytwth, choised_predict_clss, choised_target_clss = self.select_train_samples(predict_dxdydwdh_batch, predict_clss_batch, target_xyxy_batch, anchors_xyxy)
				sampled_positive_dxdydwdh.append(choised_positive_dxdydwdh)
				sampled_positive_txtytwth.append(choised_positive_txtytwth)
				sampled_predict_clss.append(choised_predict_clss)
				sampled_target_clss.append(choised_target_clss)

			sampled_positive_dxdydwdh = torch.cat(sampled_positive_dxdydwdh, dim=0)
			sampled_positive_txtytwth = torch.cat(sampled_positive_txtytwth, dim=0)
			sampled_predict_clss = torch.cat(sampled_predict_clss, dim=0)
			sampled_target_clss = torch.cat(sampled_target_clss, dim=0)

			clss_loss, bbox_loss = self.compute_loss(sampled_positive_dxdydwdh, sampled_positive_txtytwth, sampled_predict_clss, sampled_target_clss)
			losses['clss_loss'] = clss_loss
			losses['bbox_loss'] = bbox_loss
		return finall_proposals_xyxy, losses


