import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from albumentations import (
	HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
	Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
	ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from sklearn.neighbors import KNeighborsClassifier

from model_classfication.model import ResNet101, Efficient, Productnet

def Test_Transforms(img_size):
	return Compose([
        Resize(img_size[0], img_size[1]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

class Classfication():
	def __init__(self, model_weights, num_classes, input_height, input_width):
		self.model_weights = model_weights
		self.num_classes = num_classes
		self.input_height = input_height
		self.input_width = input_width

		self.test_transform = Test_Transforms((input_height, input_width))
	def get_model(self):
		model = Productnet(backbone_weights=None, num_classes=self.num_classes)
		pretrained_dict = torch.load(self.model_weights)
		single_dict = {}
		for k, v in pretrained_dict.items():
			single_dict[k[7:]] = v
		model.load_state_dict(single_dict)
		model = model.cuda()
		return model

	def padding(self, img, target_size):
		ori_height, ori_width, c = img.shape

		new_img = np.zeros([target_size, target_size, c], dtype=img.dtype)
		new_img.fill(128.0)

		scale = target_size / max(ori_height, ori_width)
		scale_height = int(scale * ori_height)
		scale_widht = int(scale * ori_width)
		scale_img = cv2.resize(img, (scale_widht, scale_height))

		new_img[:scale_height, :scale_widht, :] = scale_img

		return new_img

	def predict(self, model, all_images):
		transform_images = []
		for img in all_images:
			img = self.padding(img, self.input_height)

			img = self.test_transform(image=img)['image']

			transform_images.append(img.unsqueeze(0))
		transform_images = torch.cat(transform_images, dim=0) # torch.Size([31, 3, 224, 224])

		model.eval()
		with torch.no_grad():
			image = transform_images.cuda()
			feature = model(image.cuda())
			feature = feature.cpu().numpy()
		return feature


