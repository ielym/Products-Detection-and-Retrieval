import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math

class ArcMarginProduct(nn.Module):
	r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
	"""
	def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
		super(ArcMarginProduct, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)

		self.easy_margin = easy_margin
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m

	def forward(self, inputs, labels):
		# --------------------------- cos(theta) & phi(theta) ---------------------------
		cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
		sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
		phi = cosine * self.cos_m - sine * self.sin_m
		if self.easy_margin:
			phi = torch.where(cosine > 0, phi, cosine)
		else:
			phi = torch.where(cosine > self.th, phi, cosine - self.mm)
		# --------------------------- convert label to one-hot ---------------------------
		# one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
		one_hot = torch.zeros(cosine.size(), device=inputs.device)
		one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
		# -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
		output *= self.s
		loss = F.cross_entropy(output, labels)
		return output, loss

class ProductNet(nn.Module):
	def __init__(self, backbone):
		super(ProductNet, self).__init__()
		self.backbone = backbone

		self.arcface_loss = ArcMarginProduct(in_features=512, out_features=116, s=32, m=0.6, easy_margin=False)

	def compute_classfication_loss(self, anchor_predict, same_predict, differ_predict, label_anchor, label_same, label_differ):
		predicts = torch.cat([anchor_predict, same_predict, differ_predict], dim=0)
		targets = torch.cat([label_anchor, label_same, label_differ], dim=0)
		loss_classification = F.cross_entropy(predicts, targets.long(), reduction='mean')
		return loss_classification

	def cal_distance(self, feature1, feature2):
		# P = 2
		# D = torch.sum(torch.pow(feature1 - feature2, 2), dim=1)
		# D = torch.pow(D, 1/2)

		# cosine loss
		D = torch.sum(feature1 * feature2) / (torch.pow(torch.sum(torch.pow(feature1, 2)), 0.5) * torch.pow(torch.sum(torch.pow(feature2, 2)), 0.5))
		D = 1 - D
		return D

	def randking_loss(self, d_ap, d_an, m):
		'''
			loss = |d_ap + m - d_an| = |d_ap + max(0, m - d_an)|
		'''
		loss_ap = d_ap # tensor([14.8663, 16.9655, 16.2925, 17.4689], device='cuda:0',
		loss_an = m - d_an # tensor([29.6372, 31.9081, 31.5620, 32.3958], device='cuda:0',
		loss_an[loss_an < 0] = 0
		return torch.mean(loss_ap + loss_an)

	def triplet_loss(self, anchor_feature, positive_feature, negative_feature):
		loss = F.triplet_margin_loss(anchor_feature, positive_feature, negative_feature, margin=10, p=2, reduction='mean')
		return loss

	def compute_compare_loss(self, anchor_feature, positive_feature, negative_feature):
		# d_ap = self.cal_distance(anchor_feature, positive_feature) # torch.Size([4])
		# d_an = self.cal_distance(anchor_feature, negative_feature) # torch.Size([4])
		# loss_compare = self.randking_loss(d_ap, d_an, m=1.0)
		loss_compare = self.triplet_loss(anchor_feature, positive_feature, negative_feature)
		return loss_compare

	def forward(self, img_anchor):
	# def forward(self, img_anchor, img_same, img_differ, label_anchor, label_same, label_differ):

		anchor_predict, anchor_feature = self.backbone(img_anchor) # torch.Size([4, 116]) torch.Size([4, 2048])
		# positive_predict, positive_feature = self.backbone(img_same) # torch.Size([4, 116]) torch.Size([4, 2048])
		# negative_predict, negative_feature = self.backbone(img_differ) # torch.Size([4, 116]) torch.Size([4, 2048])

		# mean_anchor, var_anchor = torch.var_mean(anchor_feature, dim=0)
		# mean_positive, var_positive = torch.var_mean(positive_feature, dim=0)
		# mean_negative, var_negative = torch.var_mean(negative_feature, dim=0)

		# loss_classification = self.compute_classfication_loss(anchor_predict, positive_predict, negative_predict, label_anchor, label_positive, label_negative)
		# loss_compare = self.compute_compare_loss(anchor_feature, positive_feature, negative_feature)

		# loss_classfication = F.cross_entropy(anchor_predict, label_anchor.long(), reduction='mean')
		# arcface_predict, loss_compare = self.arcface_loss(anchor_feature, label_anchor.long())

		return anchor_feature
		# return anchor_predict, loss_classfication