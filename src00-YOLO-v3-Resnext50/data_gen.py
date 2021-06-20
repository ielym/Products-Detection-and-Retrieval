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
import json
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from torchtoolbox.transform import Cutout
from PIL import Image
from albumentations import (
    IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from utils.load_xml import load_voc_xml
from utils.datatransform import HorizontalFlip, VerticalFlip

class DataAugmentation():

	def Train_Transforms(self):
		return Compose([
			HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
			RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
			ToTensorV2(p=1.0),
		], p=1.)

	def Val_Transforms(self, img_size):
		pass

	def Test_Transforms(self):
		pass

def Padding(img, target_height, target_width):
	ori_h, ori_w, ori_c = img.shape
	padding_img = np.zeros(shape=(target_height, target_width, ori_c), dtype=img.dtype)
	padding_img[:ori_h, :ori_w, :] = img
	return padding_img

class BaseDataset(Dataset):
	def __init__(self, samples, input_size, max_objs, num_classes):
		self.samples = samples

		self.input_size = input_size

		self.max_objs = max_objs

		self.num_classes = num_classes

		self.augmentation = DataAugmentation()

	def __len__(self):
		return len(self.samples)

	def box_encoder(self, boxes, scale_factor):
		boxes = boxes * scale_factor
		return boxes

	def make_yolo_label(self, gt_boxes, gt_labels):
		target = np.zeros(shape=(self.max_objs, 5), dtype=np.float32)
		for idx in range(len(gt_labels)):
			box = gt_boxes[idx, :]
			label = gt_labels[idx, :]

			target[idx, 0:4] = box / self.input_size
			target[idx, 4] = label
		target = torch.from_numpy(target).float()
		return target

	def preprocess_img_label(self, img_path, gt_boxes, gt_labels):
		img = cv2.imread(img_path)

		img, gt_boxes = HorizontalFlip(img, gt_boxes, p=0.5)
		img, gt_boxes = VerticalFlip(img, gt_boxes, p=0.5)

		ori_height, ori_width, ori_channel = img.shape
		ori_longer_size = max(ori_height, ori_width)
		scale_factor = self.input_size / ori_longer_size
		target_height = int(ori_height * scale_factor)
		target_width = int(ori_width * scale_factor)

		img = cv2.resize(img, (target_width, target_height), cv2.INTER_AREA)
		img = Padding(img, self.input_size, self.input_size)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		transforms = self.augmentation.Train_Transforms()
		image = transforms(image=img)['image']

		gt_boxes = self.box_encoder(gt_boxes, scale_factor)
		targets = self.make_yolo_label(gt_boxes, gt_labels)
		return image, targets

	def get_label(self, obj):
		img_path = obj['img_name']
		img_height = obj['img_height']
		img_width = obj['img_width']

		bbox_category = obj['obj']

		gt_boxes = []
		labels = []
		for bc in bbox_category:
			x_min, y_min, w, h = bc[:4]
			x_max, y_max = x_min + w, y_min + h
			gt_boxes.append([x_min, y_min, x_max, y_max])
			labels.append(bc[4]+1)
		gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape([-1, 4])
		gt_labels = np.array(labels, dtype=np.int).reshape([-1, 1])
		return img_path, img_height, img_width, gt_boxes, gt_labels

	def __getitem__(self, idx):
		obj = self.samples[idx]

		img_path, img_height, img_width, gt_boxes, gt_labels = self.get_label(obj)
		image, targets = self.preprocess_img_label(img_path, gt_boxes, gt_labels)

		return image, targets

def data_flow(base_dir, input_size, max_objs, num_classes):

	# input_size = np.random.choice([320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
	# print('input_size : ', input_size)

	train_val_json_path = os.path.join(base_dir, 'train_ab.json')
	train_val_json = json.load(open(train_val_json_path, 'r'))

	train_val_samples = []
	for k, v in train_val_json.items():
		train_val_json[k]['img_name'] = os.path.join(base_dir, v['img_name'])
		train_val_json[k]['img_name'] = train_val_json[k]['img_name'].replace('\\', '/')
		train_val_samples.append(train_val_json[k])

	np.random.shuffle(train_val_samples)

	train_samples = train_val_samples[:int(0.95*len(train_val_samples))]
	val_samples = train_val_samples[int(0.95*len(train_val_samples)):]
	print('total samples: %d, training samples: %d, validation samples: %d' % (len(train_val_samples), len(train_samples), len(val_samples)))
	train_dataset = BaseDataset(train_samples, input_size, max_objs, num_classes)
	validation_dataset = BaseDataset(val_samples, input_size, max_objs, num_classes)

	return train_dataset, validation_dataset


if __name__ == '__main__':

	# data_flow(r'S:\DataSets\cassava-leaf-disease-classification', input_shape=(3, 500, 500), num_classes=5)
	train_dataset, validation_dataset = data_flow(base_dir=r'S:\DataSets\productsDET\V01', input_size=416, max_objs=200, num_classes=116)

	id_name_map = {0: 'asamu', 1: 'baishikele', 2: 'baokuangli', 3: 'aoliao', 4: 'bingqilinniunai', 5: 'chapai',
				   6: 'fenda', 7: 'guolicheng', 8: 'haoliyou', 9: 'heweidao', 10: 'hongniu', 11: 'hongniu2',
				   12: 'hongshaoniurou', 13: 'kafei', 14: 'kaomo_gali', 15: 'kaomo_jiaoyan', 16: 'kaomo_shaokao',
				   17: 'kaomo_xiangcon', 18: 'kele', 19: 'laotansuancai', 20: 'liaomian', 21: 'lingdukele',
				   22: 'maidong', 23: 'mangguoxiaolao', 24: 'moliqingcha', 25: 'niunai', 26: 'qinningshui',
				   27: 'quchenshixiangcao', 28: 'rousongbing', 29: 'suanlafen', 30: 'tangdaren', 31: 'wangzainiunai',
				   32: 'weic', 33: 'weitanai', 34: 'weitaningmeng', 35: 'wulongcha', 36: 'xuebi', 37: 'xuebi2',
				   38: 'yingyangkuaixian', 39: 'yuanqishui', 40: 'xuebi-b', 41: 'kebike', 42: 'tangdaren3',
				   43: 'chacui', 44: 'heweidao2', 45: 'youyanggudong', 46: 'baishikele-2', 47: 'heweidao3', 48: 'yibao',
				   49: 'kele-b', 50: 'AD', 51: 'jianjiao', 52: 'yezhi', 53: 'libaojian', 54: 'nongfushanquan',
				   55: 'weitanaiditang', 56: 'ufo', 57: 'zihaiguo', 58: 'nfc', 59: 'yitengyuan', 60: 'xianglaniurou',
				   61: 'gudasao', 62: 'buding', 63: 'ufo2', 64: 'damaicha', 65: 'chapai2', 66: 'tangdaren2',
				   67: 'suanlaniurou', 68: 'bingtangxueli', 69: 'weitaningmeng-bottle', 70: 'liziyuan', 71: 'yousuanru',
				   72: 'rancha-1', 73: 'rancha-2', 74: 'wanglaoji', 75: 'weitanai2', 76: 'qingdaowangzi-1',
				   77: 'qingdaowangzi-2', 78: 'binghongcha', 79: 'aerbeisi', 80: 'lujikafei', 81: 'kele-b-2',
				   82: 'anmuxi', 83: 'xianguolao', 84: 'haitai', 85: 'youlemei', 86: 'weiweidounai', 87: 'jindian',
				   88: '3jia2', 89: 'meiniye', 90: 'rusuanjunqishui', 91: 'taipingshuda', 92: 'yida', 93: 'haochidian',
				   94: 'wuhounaicha', 95: 'baicha', 96: 'lingdukele-b', 97: 'jianlibao', 98: 'lujiaoxiang', 99: '3+2-2',
				   100: 'luxiangniurou', 101: 'dongpeng', 102: 'dongpeng-b', 103: 'xianxiayuban', 104: 'niudufen',
				   105: 'zaocanmofang', 106: 'wanglaoji-c', 107: 'mengniu', 108: 'mengniuzaocan', 109: 'guolicheng2',
				   110: 'daofandian1', 111: 'daofandian2', 112: 'daofandian3', 113: 'daofandian4', 114: 'yingyingquqi',
				   115: 'lefuqiu'}
	epoch = 1
	while True:
		epoch += 1
		for image, target in train_dataset:
			img = image.numpy()
			target = target.numpy()
			img = np.transpose(img, [1, 2, 0])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			print(img.shape)

			objs = target[target[..., 4] > 0]
			bboxes = objs[..., :4]
			categories = objs[..., 4]
			for box, category in zip(bboxes, categories):
				box = box * 416
				x_min, y_min, x_max, y_max = box
				img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness=2)
				cv2.imshow('img', img)
				cv2.waitKey()
