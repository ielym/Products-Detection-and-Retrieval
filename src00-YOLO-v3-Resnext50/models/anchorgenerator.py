import numpy as np
from matplotlib import pyplot as plt
import os
import torch
import json

import xml.etree.ElementTree as ET

def load_voc_xml(annotation_path):
	root = ET.parse(annotation_path).getroot()
	size_node = root.find('size')

	width = int(float(size_node.find('width').text))
	height = int(float(size_node.find('height').text))
	depth = int(float(size_node.find('depth').text))

	annotation = {'size':{'height':height, 'width':width, 'depth':depth}, 'boxes':[]}
	for obj in root.findall('object'):
		temp_dict = {}

		bndbox = obj.find('bndbox')
		temp_dict['name'] = obj.find('name').text
		temp_dict['x_min'] = int(float(bndbox.find('xmin').text))
		temp_dict['y_min'] = int(float(bndbox.find('ymin').text))
		temp_dict['x_max'] = int(float(bndbox.find('xmax').text))
		temp_dict['y_max'] = int(float(bndbox.find('ymax').text))

		annotation['boxes'].append(temp_dict)
	return annotation

class KMeans():
	def __init__(self, xs, k=9):
		self.num_samples = len(xs)
		self.samples = xs
		self.k = k

		np.random.shuffle(self.samples)

		self.centers = self.samples[:k, :]

	def cal_distance(self, xs, centers):
		# 1 - iou(x, center)

		x_min = 0 - xs[:, 0] / 2
		y_min = 0 - xs[:, 1] / 2
		x_max = 0 + xs[:, 0] / 2
		y_max = 0 + xs[:, 1] / 2

		cx_min = 0 - centers[:, 0] / 2
		cy_min = 0 - centers[:, 1] / 2
		cx_max = 0 + centers[:, 0] / 2
		cy_max = 0 + centers[:, 1] / 2

		ix_min = np.maximum(x_min[:, None], cx_min)
		iy_min = np.maximum(y_min[:, None], cy_min)
		ix_max = np.minimum(x_max[:, None], cx_max)
		iy_max = np.minimum(y_max[:, None], cy_max)

		areaxs = xs[:, 1] * xs[:, 0] # (15774,)
		areacenter = centers[:, 1] * centers[:, 0] # (9,)
		areainter = np.clip((ix_max - ix_min), a_min=0, a_max=999999) * np.clip((iy_max - iy_min), a_min=0, a_max=999999) # (15774, 9)

		iou = areainter / (areaxs[:, None] + areacenter - areainter)
		return 1 - iou

	def fit(self, iters):
		for i in range(iters):
			self._fit()
		# for k in range(1, self.k+1):
		# 	plt.scatter(self.samples[self.categories==k, 0], self.samples[self.categories==k, 1], c=np.zeros_like(self.samples[self.categories==k, 0]).fill(k), alpha=0.8)
		# 	plt.scatter(self.centers[:, 0], self.centers[:, 1], c='black', marker='x')
		# plt.show()
		return self.centers

	def _fit(self):
		distance = self.cal_distance(self.samples, self.centers)
		self.categories = np.argmin(distance, axis=1) + 1

		new_centers = []
		for category_id in range(1, self.k+1):
			sample_category = self.samples[self.categories==category_id]

			if sample_category.shape[0] == 0:
				new_centers.append(self.centers[category_id-1, :])
			else:
				new_x = np.mean(sample_category[:, 0])
				new_y = np.mean(sample_category[:, 1])
				new_centers.append([new_x, new_y])
		self.centers = np.array(new_centers).reshape([-1, 2])

class ANchorGenerator():
	def __init__(self, base_dir, k=5, iters=100, mode='Train'):
		json_path = os.path.join(base_dir, 'train_ab.json')
		img_jsons = json.load(open(json_path, 'r'))
		width_height = []
		for _, v in img_jsons.items():
			objs = v['obj']
			img_height = v['img_height']
			img_width = v['img_width']
			for obj in objs:
				x_min, y_min, w, h = obj[:4]
				x_max, y_max = x_min + w, y_min + h
				sizes = [x_min/img_width, y_min/img_height, x_max/img_width, y_max/img_height]
				width_height.extend(sizes)
		width_height = np.array(width_height, dtype=np.float32).reshape([-1, 2])

		centers = KMeans(xs=width_height, k=k).fit(iters=iters)
		centers = centers.tolist()
		centers.sort(key=lambda x: x[0]*x[1])

		self.centers = np.array(centers, dtype=np.float).reshape([-1, 2])

		if mode == 'Train':
			self.save_anchors()

	def save_anchors(self):
		with open('./anchors.txt', 'w') as f:
			f.write(str(self.centers))
			print('Save anchors to anchors.txt')

	def get_anchors(self):
		return torch.from_numpy(self.centers)