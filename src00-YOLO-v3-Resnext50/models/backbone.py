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
		elif backbone_name == 'darknet19':
			from .darknet import DarkNet19
			self.feature_extract = DarkNet19()
			self.backbone_infos = {
				'in_channels' : 3,
				'out_channels' : [256, 512, 1024]
			}
			if pretrained:
				if not os.path.exists(backbone_weights):
					raise Exception('[ERROR] : pretrained_weights : {} not exist!'.format(backbone_weights))
				pretrained_weights = torch.load(backbone_weights)
				single_dict = {}
				for k, v in pretrained_weights.items():
					single_dict[k[7:]] = v
				self.feature_extract.load_state_dict(single_dict)
		elif backbone_name == 'resnet50':
			from .resnet import resnet50
			self.feature_extract = resnet50()
			self.backbone_infos = {
				'in_channels' : 3,
				'out_channels' : [512, 1024, 2048],
			}
			if pretrained:
				if not os.path.exists(backbone_weights):
					raise Exception('[ERROR] : pretrained_weights : {} not exist!'.format(backbone_weights))
				pretrained_weights = torch.load(backbone_weights)
				pretrained_weights.pop('fc.weight')
				pretrained_weights.pop('fc.bias')
				self.feature_extract.load_state_dict(pretrained_weights)
		elif backbone_name == 'resnext101_32x8d':
			from .resnet import resnext101_32x8d
			self.feature_extract = resnext101_32x8d()
			self.backbone_infos = {
				'in_channels' : 3,
				'out_channels' : [512, 1024, 2048],
			}
			if pretrained:
				if not os.path.exists(backbone_weights):
					raise Exception('[ERROR] : pretrained_weights : {} not exist!'.format(backbone_weights))
				pretrained_weights = torch.load(backbone_weights)
				pretrained_weights.pop('fc.weight')
				pretrained_weights.pop('fc.bias')
				self.feature_extract.load_state_dict(pretrained_weights)
		elif backbone_name == 'efficientnet-b7':
			from .efficientnet import EfficientNet
			self.feature_extract = EfficientNet.from_name(model_name='efficientnet-b7')
			self.backbone_infos = {
				'in_channels' : 3,
				'out_channels' : 2560,
			}
			if pretrained:
				if not os.path.exists(backbone_weights):
					raise Exception('[ERROR] : pretrained_weights : {} not exist!'.format(backbone_weights))
				pretrained_weights = torch.load(backbone_weights)
				pretrained_weights.pop('_fc.weight')
				pretrained_weights.pop('_fc.bias')
				self.feature_extract.load_state_dict(pretrained_weights)

	def forward(self, x):
		x = self.feature_extract(x)
		return x

	def backbone_feature_channels(self):
		return self.backbone_infos['out_channels']