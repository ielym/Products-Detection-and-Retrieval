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
	start = time.time()

	model = model_fn(args)

	# ==================================================================================================================
	TEST_DIR = r'S:\DataSets\productsDET\V01\test\a_images'
	json_DIR = os.path.join(r'S:\DataSets\productsDET\V01\test', 'a_annotations.json')
	test_images = glob(os.path.join(TEST_DIR, '*'))

	annotations_json = json.load(open(json_DIR, 'r'))
	annotations_images = annotations_json['images']
	img_name_id_map = {}
	img_name_size_map = {}
	for image_info in annotations_images:
		img_name_id_map[image_info['file_name']] = image_info['id']
		img_name_size_map[image_info['file_name']] = (int(image_info['height']), int(image_info['width']))

	images = []
	annotations = []
	model.eval()
	with torch.no_grad():
		for cnt, img_path in enumerate(test_images):
			print('\r {} / {}'.format(cnt+1, len(test_images)), end='')

			image, img = preprocess_img(img_path, args.input_size)
			image = image.cuda()
			predict, _, _ = model(image)
			boxes, categories, probs = post_process(predict, args, img.shape)
			boxes = boxes.cpu().numpy()
			categories = categories.cpu().numpy()
			probs = probs.cpu().numpy()

			file_name = os.path.basename(img_path)
			img_id = img_name_id_map[file_name]
			images.append({"file_name": file_name, "id": img_id})
			img_height, img_width = img_name_size_map[file_name]
			# resized_img = cv2.resize(img, (img_width, img_height))
			# print(resized_img.shape)
			for box, category, prob in zip(boxes, categories, probs):
				x_min, y_min, x_max, y_max = box
				x_min = x_min / img.shape[1] * img_width
				y_min = y_min / img.shape[0] * img_height
				x_max = x_max / img.shape[1] * img_width
				y_max = y_max / img.shape[0] * img_height
				w = x_max - x_min
				h = y_max - y_min
				single_dict = {
								"image_id": int(img_id),
								"category_id": int(category - 1),
								"bbox": [float(x_min), float(y_min), float(w), float(h)],
								"score": float(prob)
							}
				annotations.append(single_dict)
				# print((int(x_min), int(y_min)), (int(x_max), int(y_max)))
				# resized_img = cv2.rectangle(resized_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255), thickness=2)
				# print(int(category - 1))
				# cv2.imshow('img', resized_img)
				# cv2.waitKey()
		print('\n', (time.time() - start) / len(test_images), ' sec per image avg.')

	predictions = {"images": images, "annotations": annotations}
	result_json = json.dumps(predictions)
	with open("predictions.json", "w") as f:
		f.write(result_json)

	# ==================================================================================================================