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
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import torchvision
import json
import time

from models.model import FasterRcnn
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
	h, w, c = img.shape
	padded_img = np.zeros(shape=(target_height, target_width, c)).astype(img.dtype)
	padded_img[:h, :w, :c] = img
	return padded_img

def preprocess_img(img_path, input_size):
	img = cv2.imread(img_path)

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
	model = FasterRcnn(args=args)

	inference_dict = torch.load(args.inference_weights)
	single_dict = {}
	for k, v in inference_dict.items():
		single_dict[k[7:]] = v
	model.load_state_dict(single_dict)

	model = model.cuda()
	return model

def batched_nms(boxes, scores, idxs, iou_threshold):
	# max_coordinate = 9999
	# offsets = idxs.to(boxes) * max_coordinate
	# boxes_for_nms = boxes + offsets[:, None]
	keep = torchvision.ops.nms(boxes, scores, iou_threshold)
	return keep

def post_process(predicts, args, resized_img_shape):

	probs, categories, boxes = predicts

	boxes[:, 0] = boxes[:, 0].clamp(min=0, max=resized_img_shape[1]-1)
	boxes[:, 1] = boxes[:, 1].clamp(min=0, max=resized_img_shape[0]-1)
	boxes[:, 2] = boxes[:, 2].clamp(min=0, max=resized_img_shape[1]-1)
	boxes[:, 3] = boxes[:, 3].clamp(min=0, max=resized_img_shape[0]-1)

	# remove min size
	keep = torch.nonzero((boxes[:, 2]-boxes[:, 0]>=args.remove_min_size) & (boxes[:, 3]-boxes[:, 1]>=args.remove_min_size)).view(-1)
	probs = probs[keep].view(-1)
	categories = categories[keep].view(-1)
	boxes = boxes[keep, :]

	if boxes.size(0) == 0:
		empty = torch.tensor([]).to(boxes)
		return empty, empty, empty

	keep = batched_nms(boxes, probs, categories, args.nms_thresh)
	probs = probs[keep]
	categories = categories[keep]
	boxes = boxes[keep, :]
	return boxes, categories, probs

def Inference(args):
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
	colors = {0: (70, 20, 210), 1: (250, 100, 50), 2: (30, 0, 20), 3: (40, 10, 190), 4: (20, 200, 90), 5: (190, 210, 30), 6: (160, 170, 0), 7: (180, 160, 130), 8: (60, 230, 70), 9: (40, 90, 130), 10: (160, 70, 60), 11: (230, 240, 20), 12: (50, 60, 140), 13: (60, 30, 200), 14: (100, 220, 200), 15: (140, 200, 110), 16: (120, 40, 210), 17: (60, 120, 210), 18: (40, 70, 190), 19: (150, 230, 110), 20: (50, 110, 240), 21: (130, 150, 160), 22: (240, 190, 70), 23: (230, 10, 140), 24: (140, 10, 200), 25: (120, 0, 30), 26: (250, 210, 230), 27: (220, 210, 130), 28: (120, 140, 170), 29: (190, 70, 0), 30: (180, 160, 30), 31: (0, 80, 110), 32: (20, 10, 80), 33: (220, 160, 190), 34: (100, 240, 0), 35: (210, 80, 10), 36: (30, 70, 120), 37: (220, 110, 250), 38: (130, 50, 0), 39: (240, 120, 70), 40: (190, 140, 20), 41: (110, 130, 60), 42: (220, 30, 80), 43: (100, 150, 0), 44: (80, 0, 20), 45: (130, 170, 160), 46: (190, 220, 190), 47: (40, 10, 80), 48: (130, 250, 10), 49: (170, 160, 80), 50: (190, 60, 70), 51: (250, 30, 150), 52: (140, 80, 150), 53: (30, 160, 230), 54: (150, 170, 220), 55: (20, 40, 160), 56: (210, 60, 0), 57: (210, 250, 0), 58: (160, 220, 180), 59: (120, 110, 10), 60: (10, 130, 20), 61: (30, 200, 130), 62: (110, 90, 30), 63: (180, 30, 230), 64: (110, 70, 240), 65: (50, 250, 110), 66: (250, 140, 190), 67: (250, 210, 140), 68: (90, 160, 30), 69: (190, 110, 100), 70: (110, 20, 180), 71: (120, 100, 100), 72: (10, 100, 250), 73: (90, 220, 110), 74: (140, 160, 170), 75: (170, 90, 160), 76: (80, 10, 100), 77: (160, 120, 250), 78: (30, 190, 90), 79: (210, 80, 50), 80: (150, 180, 20), 81: (130, 120, 230), 82: (220, 10, 120), 83: (170, 100, 150), 84: (120, 180, 60), 85: (230, 170, 130), 86: (30, 30, 200), 87: (70, 40, 190), 88: (240, 120, 20), 89: (210, 140, 120), 90: (140, 100, 110), 91: (70, 190, 120), 92: (180, 210, 60), 93: (70, 100, 150), 94: (120, 230, 80), 95: (240, 170, 250), 96: (60, 220, 170), 97: (100, 70, 40), 98: (190, 60, 210), 99: (200, 150, 110), 100: (130, 190, 0), 101: (110, 130, 190), 102: (250, 70, 90), 103: (140, 50, 90), 104: (90, 210, 0), 105: (170, 200, 130), 106: (130, 70, 90), 107: (30, 210, 80), 108: (190, 150, 20), 109: (140, 120, 200), 110: (230, 60, 70), 111: (80, 240, 220), 112: (230, 30, 110), 113: (60, 240, 20), 114: (70, 190, 150), 115: (210, 210, 40)}

	model = model_fn(args)

	TEST_DIR = r'S:\DataSets\productsDET\V01\test\a_images'
	# TEST_DIR = r'S:\DataSets\productsDET\V01\test\b_images'
	# TEST_DIR = r'S:\DataSets\productsDET\V01\train\a_images'
	# TEST_DIR = r'S:\DataSets\productsDET\V01\train\b_images'
	# TEST_DIR = r'S:\DataSets\productsDET\temp'
	test_images = glob(os.path.join(TEST_DIR, '*'))
	np.random.shuffle(test_images)

	model.eval()
	with torch.no_grad():
		for img_path in test_images:
			stime = time.time()

			image, img = preprocess_img(img_path, args.input_size)
			image = image.cuda()
			predict, _, _ = model(image)
			boxes, categories, probs = post_process(predict, args, img.shape)
			boxes = boxes.cpu().numpy()
			categories = categories.cpu().numpy() - 1
			probs = probs.cpu().numpy()

			print(time.time() - stime)
			show_img = img.copy()
			show_img = cv2.resize(show_img, (show_img.shape[1] * 2, show_img.shape[0] * 2))
			for box, category, prob in zip(boxes, categories, probs):
				box = box * 2
				show_img = cv2.rectangle(show_img, (box[0], box[1]), (box[2], box[3]), color= colors[category], thickness=2)
				show_img = cv2.rectangle(show_img, (box[0], box[1]), (int(box[0] + 128), int(box[1] - 32)), color=colors[category], thickness=-1)
				show_img = cv2ImgAddText(show_img, '\t' + str(id_name_map[category]), box[0] - 24, box[1] - 24, textColor=(255, 255, 255), textSize=24)
			cv2.imshow('show_img', show_img)
			cv2.waitKey()