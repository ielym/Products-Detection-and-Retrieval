import torchvision
import torch
import torch.nn as nn
import collections
import cv2
import numpy as np
from .resnext101_32x8d import resnext101_32x8d, resnet50
from .efficientnet import EfficientNet
from .productnet import ProductNet

def _resnet101(weights, num_classes):
	model = resnext101_32x8d(pretrained=False, num_classes=num_classes)
	if weights:
		pretrained_dict = torch.load(weights)
		model.load_state_dict(pretrained_dict, strict=False)
	return model

def ResNet101(weights, input_shape, num_classes):
	model = _resnet101(weights, num_classes)
	return model

def _efficientnet(model_name, weights, num_classes):
	model = EfficientNet.from_name(model_name=model_name, override_params={'num_classes': num_classes})
	if weights:
		pretrained_dict = torch.load(weights)
		pretrained_dict.pop('_fc.weight')
		pretrained_dict.pop('_fc.bias')
		model.load_state_dict(pretrained_dict, strict=False)
	return model

def Efficient(model_name, weights, input_shape, num_classes):
	model = _efficientnet(model_name, weights, num_classes)
	return model

def Productnet(backbone_weights, num_classes):
	backbone = _resnet101(backbone_weights, num_classes)
	model = ProductNet(backbone)
	return model

if __name__ == '__main__':
	from torchsummary import summary
	model = ResNet101(weights=None, input_shape=(3, 224, 224), num_classes=3)
	# model = Efficient(model_name='efficientnet-b7', weights='./zoo/efficientnet-b7-dcc49843.pth', input_shape=(3, 224, 224), num_classes=3)

	summary(model, (3, 224, 224), device='cpu')

# import tensorwatch as tw
	# img = torch.ones(size=(1, 3, 224, 224))
	# tw.draw_model(model, (1, 3, 224, 224), png_filename='./res.png')
