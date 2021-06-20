import cv2
import os
from glob import glob
import numpy as np
import json

coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

base_dir = r'S:\DataSets\coco2017'

train_img_dir = os.path.join(base_dir, 'train2017')
val_img_dir = os.path.join(base_dir, 'val2017')

annotation_dir = os.path.join(base_dir, 'annotations')
train_annotation_path = os.path.join(annotation_dir, 'instances_train2017.json')
val_annotation_path = os.path.join(annotation_dir, 'instances_val2017.json')

train_json = json.load(open(train_annotation_path, 'r')) # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
val_json = json.load(open(val_annotation_path, 'r')) # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

images = val_json['images'] # <class 'list'>
annotations = val_json['annotations'] # <class 'list'>

results = {}
for image in images:
	img_id = image['id']
	img_name = image['file_name']
	img_height = image['height']
	img_width = image['width']
	results[img_id] = {"img_name":img_name, "img_height":img_height, "img_width":img_width, "obj":[]}

for annotation in annotations:
	img_id = annotation['image_id']
	category = annotation['category_id']
	obj = annotation['bbox']
	obj.append(category)
	results[img_id]['obj'].append(obj)

with open(r'./val.json', 'w') as f:
	json.dump(results, f)

for img_id, img_infos in results.items():
	img_name = img_infos['img_name']
	img_height = img_infos['img_height']
	img_width = img_infos['img_width']

	img = cv2.imread(os.path.join(val_img_dir, img_name))
	obj = img_infos['obj']
	for o in obj:
		box = o[:4]
		x_min, y_min, w, h = box
		x_max = x_min + w
		y_max = y_min + h

		category = o[4]
		img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255), thickness=2)
		print(coco_id_name_map[category])
		cv2.imshow('img', img)
		cv2.waitKey()
