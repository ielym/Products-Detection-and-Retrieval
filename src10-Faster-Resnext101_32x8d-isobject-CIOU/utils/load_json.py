import json
import numpy as np
import os

def preprocess(json_content, train_dir):
	images = json_content['images'] # [{'file_name': 'ori_XYGOC20200917193630674-3_0.jpg', 'height': 720, 'width': 960, 'id': 0}]
	annotations = json_content['annotations'] # {'area': 43560, 'iscrowd': 0, 'image_id': 1190, 'bbox': [551, 255, 198, 220], 'category_id': 19, 'id': 13075}

	results = {}
	for image in images:
		img_id = image['id']
		img_name = os.path.join(train_dir, image['file_name'])
		img_height = image['height']
		img_width = image['width']
		results[img_id] = {"img_name":img_name, "img_height":img_height, "img_width":img_width, "obj":[]}

	for annotation in annotations:
		img_id = annotation['image_id']
		category = annotation['category_id']
		obj = annotation['bbox']
		obj.append(category)
		results[img_id]['obj'].append(obj)
	return results

def get_annotation(json_path, train_dir):
	json_content = json.load(open(json_path, 'r'))
	resutls = preprocess(json_content, train_dir)
	return resutls

if __name__ == '__main__':
	json_path = r'S:\DataSets\productsDET\V01\train\a_annotations.json'
	train_dir = r'S:\DataSets\productsDET\V01\train\a_images'

	resutls = get_annotation(json_path, train_dir)

	print(resutls.keys())
	print(len(resutls.keys()))