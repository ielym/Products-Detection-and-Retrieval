# -*- coding: utf-8 -*-
import torch
import torchvision
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
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import time
import json
from models.model import YOLO

from utils.showimg import cv2ImgAddText

class DataAugmentation():

	def Train_Transforms(self):
		return Compose([
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

def preprocess_img(img, input_size):

	ori_height, ori_width, ori_channel = img.shape
	scale_factor = input_size / max(ori_height, ori_width)
	target_height = int(ori_height * scale_factor)
	target_width = int(ori_width * scale_factor)

	img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
	image = Padding(img, input_size, input_size)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	augmentation = DataAugmentation()
	transforms = augmentation.Train_Transforms()
	image = transforms(image=image)['image']

	image = image.unsqueeze(0)
	return image, img

def model_fn(args):

	yolo_params = {
		'mode' : args.mode,
		'num_anchors_per_level' : args.num_anchors_per_level,
		'bg_threshold' : args.bg_threshold,
		'num_classes' : args.num_classes,
		'data_local' : args.data_local,

		'prob_threshold' : args.prob_threshold,
	}

	model = YOLO(
					backbone_name=args.backbone_name,
					backbone_weights = args.backbone_weights,
					backbone_pretrained=args.backbone_pretrained,

					yolo_params = yolo_params,
				)

	inference_dict = torch.load(args.inference_weights)
	single_dict = {}
	for k, v in inference_dict.items():
		single_dict[k[7:]] = v
	model.load_state_dict(single_dict)

	model = model.cuda()
	return model

def batched_nms(boxes, scores, idxs, iou_threshold):
	max_coordinate = 9999
	offsets = idxs.to(boxes) * max_coordinate
	boxes_for_nms = boxes + offsets[:, None]
	keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
	return keep

def post_process(predicts, args, resized_img_shape):

	probs, categories, boxes = predicts

	boxes[:, 0] = boxes[:, 0].clamp(min=0, max=resized_img_shape[1]-1)
	boxes[:, 1] = boxes[:, 1].clamp(min=0, max=resized_img_shape[0]-1)
	boxes[:, 2] = boxes[:, 2].clamp(min=0, max=resized_img_shape[1]-1)
	boxes[:, 3] = boxes[:, 3].clamp(min=0, max=resized_img_shape[0]-1)

	# remove min size
	keep = torch.nonzero((boxes[:, 2]-boxes[:, 0]>=args.min_size) & (boxes[:, 3]-boxes[:, 1]>=args.min_size)).view(-1)
	probs = probs[keep].view(-1)
	categories = categories[keep].view(-1)
	boxes = boxes[keep, :]

	if boxes.size(0) == 0:
		empty = torch.tensor([]).to(boxes)
		return empty, empty, empty

	keep = batched_nms(boxes, probs, categories, args.nms_threshold)
	probs = probs[keep]
	categories = categories[keep]
	boxes = boxes[keep, :]

	return boxes, categories, probs

def Inference(args):

	model = model_fn(args)

	# TEST_DIR = r'S:\DataSets\PascalVOC2012\VOC2012test\VOCdevkit\VOC2012\JPEGImages'
	# TEST_DIR = r'E:\smoking_calling\V20\trainV20\calling'
	# TEST_DIR = r'S:\DataSets\PascalVOC2012\VOCdevkit\VOC2012\JPEGImages'
	TEST_DIR = r'S:\tempsets'
	test_images = glob(os.path.join(TEST_DIR, '*'))

	label_dict = json.load(open(r'S:\DataSets\PascalVOC2012\VOCdevkit\VOC2012\pascal_voc_classes.json', 'r'))
	reversed_label_dict = {}
	for k, v in label_dict.items():
		reversed_label_dict[v] = k
	colors = {
		1 : (128, 0, 0),
		2 : (0, 128, 0),
		3 : (128, 128, 0),
		4 : (128, 128, 128),
		5 : (128, 0, 128),
		6 : (0, 128, 128),
		7 : (0, 0, 128),
		8 : (64, 0, 0),
		9 : (192, 0, 0),
		10 : (64, 128, 0),
		11 : (192, 128, 0),
		12 : (64, 0, 128),
		13 : (192, 0, 128),
		14 : (64, 128, 128),
		15 : (192, 128, 128),
		16 : (0, 64, 0),
		17 : (128, 64, 0),
		18 : (0, 192, 0),
		19 : (128, 192, 0),
		20 : (0, 64, 128),
	}

	cap = cv2.VideoCapture(r'S:\DataSets\20210430_20210430165207_20210430170258_165222.mp4')
	# cap = cv2.VideoCapture('https://vd2.bdstatic.com/mda-kddcp6yekgh8k35f/v2-custom/sc/mda-kddcp6yekgh8k35f.mp4?v_from_s=gz_haokan_4469&auth_key=1620455316-0-0-f04e2856ea7e6dc544bb8f97d5208b32&bcevod_channel=searchbox_feed&pd=1&pt=3&abtest=')
	# cap = cv2.VideoCapture(0)
	model.eval()
	with torch.no_grad():
		while True:
			ret, frame = cap.read()
			stime = time.time()
			image, img = preprocess_img(frame, args.input_size)
			image = image.cuda()
			predict, _ = model(image)

			boxes, categories, probs = post_process(predict, args, img.shape)
			boxes = boxes.cpu().numpy()
			categories = categories.cpu().numpy() + 1
			probs = probs.cpu().numpy()
			print(time.time() - stime)
			for box, category, prob in zip(boxes, categories, probs):
				img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=colors[category], thickness=2)
				# img = cv2.rectangle(img, (box[0], box[1]), (int(box[0] + 128), int(box[1] - 32)), color=colors[category], thickness=-1)
				img = cv2ImgAddText(img, '\t' + str(reversed_label_dict[category]), box[0], box[1]-32, textColor=colors[category][::-1], textSize=20)
			cv2.imshow('img', img)
			cv2.waitKey(1)