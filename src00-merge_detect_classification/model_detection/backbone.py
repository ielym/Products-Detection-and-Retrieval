import torch
from torch import nn as nn
import torchvision
import os

class BackBone(nn.Module):
	def __init__(self, backbone_name, backbone_weights, pretrained=False):
		super(BackBone, self).__init__()

		if backbone_name == 'vgg16':
			from .vgg16 import VGG16
			self.feature_extract = VGG16()
			self.backbone_infos = {
				'in_channels' : 3,
				'out_channels' : 512,
			}
			if pretrained:
				if not os.path.exists(backbone_weights):
					raise Exception('[ERROR] : pretrained_weights : {} not exist!'.format(backbone_weights))
				weights = [v for v in torch.load(backbone_weights).values()]
				myWeights = {}
				for k in self.feature_extract.state_dict().keys():
					myWeights[k] = weights.pop(0)
				self.feature_extract.load_state_dict(myWeights)
		elif backbone_name == 'resnet50':
			from .resnet import resnet50
			self.feature_extract = resnet50()
			self.backbone_infos = {
				'in_channels' : 3,
				'out_channels' : 1024,
				'last_channels' : 2048,
			}
			if pretrained:
				if not os.path.exists(backbone_weights):
					raise Exception('[ERROR] : pretrained_weights : {} not exist!'.format(backbone_weights))
				pretrained_weghts = torch.load(backbone_weights)
				pretrained_weghts.pop('fc.weight')
				pretrained_weghts.pop('fc.bias')
				self.feature_extract.load_state_dict(pretrained_weghts)

			# self.last_layers = torch.nn.Sequential(*(list(self.feature_extract.children()))[-1:])
			# self.feature_extract = torch.nn.Sequential(*(list(self.feature_extract.children()))[:-1])
		elif backbone_name == 'resnext101_32x8d':
			from .resnet import resnext101_32x8d
			self.feature_extract = resnext101_32x8d()
			self.backbone_infos = {
				'in_channels' : 3,
				'out_channels' : 1024,
				'last_channels': 2048,
			}
			if pretrained:
				if not os.path.exists(backbone_weights):
					raise Exception('[ERROR] : pretrained_weights : {} not exist!'.format(backbone_weights))
				pretrained_weghts = torch.load(backbone_weights)
				pretrained_weghts.pop('fc.weight')
				pretrained_weghts.pop('fc.bias')
				self.feature_extract.load_state_dict(pretrained_weghts)
	def forward(self, x):
		x = self.feature_extract(x)
		return x

	def backbone_feature_channels(self):
		return self.backbone_infos['out_channels']

	def backbone_last_channels(self):
		return self.backbone_infos['last_channels']
