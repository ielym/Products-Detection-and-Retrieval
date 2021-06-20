# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import math
import codecs
import random
import numpy as np
import cv2
from albumentations import (
    IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import torchvision

from model_detection.model import FasterRcnn

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

def batched_nms(boxes, scores, idxs, iou_threshold):
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


class Detection():
	def __init__(self, args):
		self.args = args

		augmentation = DataAugmentation()
		self.transforms = augmentation.Train_Transforms()

	def get_model(self):
		model = FasterRcnn(args=self.args)
		inference_dict = torch.load(self.args.inference_weights)
		single_dict = {}
		for k, v in inference_dict.items():
			single_dict[k[7:]] = v
		model.load_state_dict(single_dict)

		model = model.cuda()
		return model

	def preprocess_img(self, rgb_np_img):
		input_size = self.args.input_size

		ori_height, ori_width, ori_channel = rgb_np_img.shape
		scale_factor = input_size / max(ori_height, ori_width)
		target_height = int(ori_height * scale_factor)
		target_width = int(ori_width * scale_factor)
		target_height = 32 * math.ceil(target_height / 32)
		target_width = 32 * math.ceil(target_width / 32)
		img = cv2.resize(rgb_np_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

		# image = Padding(img, input_size, input_size)

		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
		img = cv2.filter2D(img, -1, kernel=kernel)

		image = self.transforms(image=img)['image']

		image = image.unsqueeze(0)
		return image, img

	def predict(self, model, rgb_np_img):
		image, img = self.preprocess_img(rgb_np_img)
		ori_height, ori_width, _ = rgb_np_img.shape
		resized_height, resized_width, _ = img.shape

		image = image.cuda()
		predict, _, _ = model(image)
		boxes, categories, probs = post_process(predict, self.args, img.shape)
		boxes = boxes.cpu().numpy()
		probs = probs.cpu().numpy()

		boxes[..., 0] = boxes[..., 0] / resized_width * ori_width
		boxes[..., 1] = boxes[..., 1] / resized_height * ori_height
		boxes[..., 2] = boxes[..., 2] / resized_width * ori_width
		boxes[..., 3] = boxes[..., 3] / resized_height * ori_height

		return boxes, probs