import cv2
import numpy as np
from PIL import Image
from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt

def drawheatmap(img, featuremap, threshold):

	'''
	:param img: OpenCV format with shape (224, 224, 3) and pixels level between [0, 1]
	:param featuremap: numpy format with shape (224, 224, 1) and pixels level between [0, 1]
	:param threshold:
	:return: OpenCV format with shape (224, 224, 3) and pixels level between [0, 255]
	'''

	img = img * 255
	featuremap = featuremap * 255
	img = img.astype(np.uint8)
	featuremap = featuremap.astype(np.uint8)
	data = []
	h, w, c = featuremap.shape
	for y in range(h):
		for x in range(w):
			if featuremap[y, x, 0] >= threshold:
				data.append([x, y])
	background = Image.new("RGB", size=(img.shape[0], img.shape[1]), color=0)
	hm = HeatMap(data)
	hit_img = hm.heatmap(base=background, r=20)
	hit_img = cv2.cvtColor(np.asarray(hit_img), cv2.COLOR_RGB2BGR)

	overlay = img.copy()
	alpha = 0.3
	cv2.rectangle(overlay, (0, 0), (img.shape[0], img.shape[1]), (255, 0, 0), -1)
	img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
	img = cv2.addWeighted(hit_img, alpha, img, 1-alpha, 0)
	return img
