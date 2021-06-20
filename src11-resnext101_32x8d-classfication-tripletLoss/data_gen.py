# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import math
import codecs
import random
import numpy as np
from glob import glob
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from torchtoolbox.transform import Cutout
from PIL import Image
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

class DataAugmentation():

	def Train_Transforms(self, img_size):
		return Compose([
			# RandomResizedCrop(img_size[0], img_size[1]),0
			Resize(img_size[0], img_size[1], p=1),
			Transpose(p=0.5),
			HorizontalFlip(p=0.5),
			VerticalFlip(p=0.5),
			ShiftScaleRotate(p=0.5),
			HueSaturationValue(hue_shift_limit=(-1, 1), sat_shift_limit=(-1, 1), val_shift_limit=(-1, 1), p=0.5),
			RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
			CoarseDropout(p=0.5),
			Cutout(p=0.5, max_h_size=5, max_w_size=5),
			ToTensorV2(p=1.0),
		], p=1.)

	def Val_Transforms(self, img_size):
		return Compose([
			Resize(img_size[0], img_size[1]),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
			ToTensorV2(p=1.0),
		], p=1.)

	def Test_Transforms(self):
		pass

class BaseDataset(Dataset):
	def __init__(self, samples, store_samples, input_size):
		self.samples = samples
		self.store_samples = store_samples

		self.store_samples_dict = {}

		self.input_size = input_size

		self.augmentation = DataAugmentation()
		self.train_transforms = self.augmentation.Train_Transforms(img_size=(self.input_size, self.input_size))

		self.make_category_dict()

	def make_category_dict(self):
		for line in self.store_samples:
			img_path = line[0]
			category = int(line[1])

			if not category in self.store_samples_dict.keys():
				self.store_samples_dict[category] = [img_path]
			else:
				self.store_samples_dict[category].append(img_path)

	def __len__(self):
		return len(self.samples)

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

	def preprocess_img(self, img_path):
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img = self.padding(img, self.input_size)

		img = self.train_transforms(image=img)['image']
		return img

	def preprocess_label(self, label):
		label = torch.tensor(label, dtype=torch.float32)
		# label = torch.tensor(label, dtype=torch.float32).view(1, -1)
		return label

	def __getitem__(self, idx):

		anchor_info = self.samples[idx]
		anchor_path = anchor_info[0]
		anchor_category = int(anchor_info[1])

		# positive_category = anchor_category
		# positive_path = random.choice(self.store_samples_dict[positive_category])
		# negative_category = random.choice(list(self.store_samples_dict.keys()))
		# while negative_category == anchor_category:
		# 	negative_category = random.choice(list(self.store_samples_dict.keys()))
		# negative_path = random.choice(self.store_samples_dict[negative_category])

		img_anchor = self.preprocess_img(anchor_path)
		# img_positive = self.preprocess_img(positive_path)
		# img_negative = self.preprocess_img(negative_path)

		label_anchor = self.preprocess_label(anchor_category)
		# label_positive = self.preprocess_label(positive_category)
		# label_negative = self.preprocess_label(negative_category)

		# return img_anchor, img_positive, img_negative, label_anchor, label_positive, label_negative
		return img_anchor, label_anchor

def data_flow(base_dir, input_size):

	train_dir = os.path.join(base_dir, 'train', 'a_images')
	store_dir = os.path.join(base_dir, 'train', 'b_images')

	train_img_names = os.listdir(train_dir)
	store_img_names = os.listdir(store_dir)

	store_categories = set() # 104
	stroe_samples = []
	for store_img_name in store_img_names:
		img_path = os.path.join(store_dir, store_img_name)
		img_category = os.path.basename(store_img_name).split('.')[0].split('-')[1]
		stroe_samples.append((img_path, img_category))
		store_categories.add(int(img_category))
	stroe_samples = np.array(stroe_samples).reshape([-1, 2]) # (3965, 2)

	train_samples = []
	for train_img_name in train_img_names:
		img_path = os.path.join(train_dir, train_img_name)
		img_category = os.path.basename(train_img_name).split('.')[0].split('-')[1]
		if not int(img_category) in store_categories:
			continue
		train_samples.append((img_path, img_category))
	train_pathes_categories = np.array(train_samples).reshape([-1, 2]) # (23617, 2)

	# merge_pathes_categories = np.vstack([train_pathes_categories, store_pathes_categories]) # (27582, 2)
	train_pathes = stroe_samples[..., 0]
	train_categories = stroe_samples[..., 1]

	k = 0
	K_train_indexs = []
	K_test_indexs = []
	ss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=2022)
	for train_index, test_index in ss.split(train_pathes, train_categories):
		K_train_indexs.append(train_index)
		K_test_indexs.append(test_index)
	train_paths, test_paths = train_pathes[K_train_indexs[k]], train_pathes[K_test_indexs[k]]
	train_labels, test_labels = train_categories[K_train_indexs[k]], train_categories[K_test_indexs[k]]

	train_samples = np.hstack([train_paths.reshape([-1, 1]), train_labels.reshape([-1, 1])])
	val_samples = np.hstack([test_paths.reshape([-1, 1]), test_labels.reshape([-1, 1])])

	print('total samples: %d, training samples: %d, validation samples: %d' % (len(train_samples) + len(val_samples), len(train_samples), len(val_samples)))
	print('store samples : %d' % len(stroe_samples))

	train_dataset = BaseDataset(train_samples, stroe_samples, input_size)
	validation_dataset = BaseDataset(val_samples, stroe_samples, input_size)

	return train_dataset, validation_dataset


if __name__ == '__main__':

	# data_flow(r'S:\DataSets\cassava-leaf-disease-classification', input_shape=(3, 500, 500), num_classes=5)
	train_dataset, validation_dataset = data_flow(r'S:\DataSets\productsDET\V02', input_size=500)
	epoch = 1
	while True:
		epoch += 1
		# for img_anchor, img_same, img_differ, label_anchor, label_same, label_differ in train_dataset:
		for img_anchor, label_anchor in train_dataset:
			img_anchor = img_anchor.numpy()
			# img_same = img_same.numpy()
			# img_differ = img_differ.numpy()
			label_anchor = label_anchor.numpy()
			# label_same = label_same.numpy()
			# label_differ = label_differ.numpy()

			img_anchor = np.transpose(img_anchor, [1, 2, 0])
			# img_same = np.transpose(img_same, [1, 2, 0])
			# img_differ = np.transpose(img_differ, [1, 2, 0])
			img_anchor = cv2.cvtColor(img_anchor, cv2.COLOR_BGR2RGB)
			# img_same = cv2.cvtColor(img_same, cv2.COLOR_BGR2RGB)
			# img_differ = cv2.cvtColor(img_differ, cv2.COLOR_BGR2RGB)
			cv2.imshow('anchor-' + str(label_anchor), img_anchor)
			# cv2.imshow('same-' + str(label_same), img_same)
			# cv2.imshow('differ' + str(label_differ), img_differ)
			cv2.waitKey()
			cv2.destroyAllWindows()
