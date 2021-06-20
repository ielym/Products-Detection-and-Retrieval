import torch
import torchvision
import math

def trans_delta2box(deltas, anchors):
	'''
	:param deltas:  torch.Size([2, 24192, 4])
	:param anchors: torch.Size([24192, 4])
	:return:
	'''
	dx = deltas[:, :, 0]
	dy = deltas[:, :, 1]
	dw = deltas[:, :, 2]
	dh = deltas[:, :, 3]

	pw = anchors[:, 2] - anchors[:, 0]
	ph = anchors[:, 3] - anchors[:, 1]
	px = anchors[:, 0] + pw * 0.5
	py = anchors[:, 1] + ph * 0.5

	dw = torch.clamp(dw, max=math.log(1000. / 32))
	dh = torch.clamp(dh, max=math.log(1000. / 32))

	Gx = pw * dx + px
	Gy = ph * dy + py
	Gw = torch.exp(dw) * pw
	Gh = torch.exp(dh) * ph

	boxes = deltas.clone()
	boxes[:, :, 0] = Gx - 0.5 * Gw
	boxes[:, :, 1] = Gy - 0.5 * Gh
	boxes[:, :, 2] = Gx + 0.5 * Gw
	boxes[:, :, 3] = Gy + 0.5 * Gh

	return boxes

def trans_box2delta(anchors, gt_boxes):
	pw = anchors[:, 2] - anchors[:, 0]
	ph = anchors[:, 3] - anchors[:, 1]
	px = anchors[:, 0] + pw * 0.5
	py = anchors[:, 1] + ph * 0.5

	Gw = gt_boxes[:, 2] - gt_boxes[:, 0]
	Gh = gt_boxes[:, 3] - gt_boxes[:, 1]
	Gx = gt_boxes[:, 0] + Gw * 0.5
	Gy = gt_boxes[:, 1] + Gh * 0.5

	deltas = anchors.clone()
	deltas[:, 0] = (Gx - px) / pw
	deltas[:, 1] = (Gy - py) / ph
	deltas[:, 2] = torch.log(Gw / pw + 1e-8)
	deltas[:, 3] = torch.log(Gh / ph + 1e-8)

	return deltas
