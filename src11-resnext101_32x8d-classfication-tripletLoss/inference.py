import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from albumentations import (
	HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
	Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
	ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

from utils.cam import CAM
from models.model import ResNet101, Efficient

def model_fn(inference_weights):
	# model = ResNet101(weights=None, input_shape=(3, 448, 448), num_classes=5)
	model = Efficient(model_name='efficientnet-b5', weights=r'./models/zoo/efficient-b5-ns.pth', input_shape=(3, 456, 456), num_classes=1000)

	# pretrained_dict = torch.load(inference_weights)
	# single_dict = {}
	# for k, v in pretrained_dict.items():
	# 	single_dict[k[7:]] = v
	# model.load_state_dict(single_dict)
	return model

def Test_Transforms(img_size):
	return Compose([
        Resize(img_size[0], img_size[1]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

model = model_fn(r'./models/ep00056-val_acc@1_88.5460-val_lossFocalCosine_0.1470.pth')
model = model.cuda()

TEST_DIR = r'S:\DataSets\cassava-leaf-disease-classification\train_images'
test_images = os.listdir(TEST_DIR)

labels = pd.read_csv(r'S:\DataSets\cassava-leaf-disease-classification\train.csv').to_numpy()
labels_dict = {}
for line in labels:
	labels_dict[line[0]] = line[1]

img_size = (456, 456)
test_transform = Test_Transforms(img_size)

predictions = []
model.eval()
with torch.no_grad():
	for image_name in test_images:
		ori_img = cv2.imread(os.path.join(TEST_DIR, image_name))
		img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
		img = test_transform(image=img)['image']
		image = torch.unsqueeze(img, dim=0)
		target = torch.tensor([labels_dict[image_name]], dtype=torch.float32)
		image = image.cuda()
		target = target.cuda()

		out, _ = model(image.cuda())
		out = out.cpu().numpy()
		category = np.argmax(out, axis=1)


		ori_img = cv2.resize(ori_img, (img_size))
		heat_map = CAM(image, target, model, ori_img, threshold=30)

		cv2.imshow('{} - {}'.format(image_name, category), ori_img)
		cv2.imshow('heat_map', heat_map)
		cv2.waitKey()

		predictions.extend(category)

sub = pd.DataFrame({'image_id': test_images, 'label': predictions})
sub.to_csv('submission.csv', index=False)