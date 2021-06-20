import json
import numpy as np
import os
import cv2

id_name_map = {0: 'asamu', 1: 'baishikele', 2: 'baokuangli', 3: 'aoliao', 4: 'bingqilinniunai', 5: 'chapai', 6: 'fenda', 7: 'guolicheng', 8: 'haoliyou', 9: 'heweidao', 10: 'hongniu', 11: 'hongniu2', 12: 'hongshaoniurou', 13: 'kafei', 14: 'kaomo_gali', 15: 'kaomo_jiaoyan', 16: 'kaomo_shaokao', 17: 'kaomo_xiangcon', 18: 'kele', 19: 'laotansuancai', 20: 'liaomian', 21: 'lingdukele', 22: 'maidong', 23: 'mangguoxiaolao', 24: 'moliqingcha', 25: 'niunai', 26: 'qinningshui', 27: 'quchenshixiangcao', 28: 'rousongbing', 29: 'suanlafen', 30: 'tangdaren', 31: 'wangzainiunai', 32: 'weic', 33: 'weitanai', 34: 'weitaningmeng', 35: 'wulongcha', 36: 'xuebi', 37: 'xuebi2', 38: 'yingyangkuaixian', 39: 'yuanqishui', 40: 'xuebi-b', 41: 'kebike', 42: 'tangdaren3', 43: 'chacui', 44: 'heweidao2', 45: 'youyanggudong', 46: 'baishikele-2', 47: 'heweidao3', 48: 'yibao', 49: 'kele-b', 50: 'AD', 51: 'jianjiao', 52: 'yezhi', 53: 'libaojian', 54: 'nongfushanquan', 55: 'weitanaiditang', 56: 'ufo', 57: 'zihaiguo', 58: 'nfc', 59: 'yitengyuan', 60: 'xianglaniurou', 61: 'gudasao', 62: 'buding', 63: 'ufo2', 64: 'damaicha', 65: 'chapai2', 66: 'tangdaren2', 67: 'suanlaniurou', 68: 'bingtangxueli', 69: 'weitaningmeng-bottle', 70: 'liziyuan', 71: 'yousuanru', 72: 'rancha-1', 73: 'rancha-2', 74: 'wanglaoji', 75: 'weitanai2', 76: 'qingdaowangzi-1', 77: 'qingdaowangzi-2', 78: 'binghongcha', 79: 'aerbeisi', 80: 'lujikafei', 81: 'kele-b-2', 82: 'anmuxi', 83: 'xianguolao', 84: 'haitai', 85: 'youlemei', 86: 'weiweidounai', 87: 'jindian', 88: '3jia2', 89: 'meiniye', 90: 'rusuanjunqishui', 91: 'taipingshuda', 92: 'yida', 93: 'haochidian', 94: 'wuhounaicha', 95: 'baicha', 96: 'lingdukele-b', 97: 'jianlibao', 98: 'lujiaoxiang', 99: '3+2-2', 100: 'luxiangniurou', 101: 'dongpeng', 102: 'dongpeng-b', 103: 'xianxiayuban', 104: 'niudufen', 105: 'zaocanmofang', 106: 'wanglaoji-c', 107: 'mengniu', 108: 'mengniuzaocan', 109: 'guolicheng2', 110: 'daofandian1', 111: 'daofandian2', 112: 'daofandian3', 113: 'daofandian4', 114: 'yingyingquqi', 115: 'lefuqiu'}

base_dir = r'S:\DataSets\productsDET\V01'

# ===========================  Train Paths ===============================
train_dir_a = os.path.join(base_dir, r'train\a_images')
train_annotation_path_a = os.path.join(base_dir, r'train\a_annotations.json')
train_dir_b = os.path.join(base_dir, r'train\b_images')
train_annotation_path_b = os.path.join(base_dir, r'train\b_annotations.json')
train_json_a = json.load(open(train_annotation_path_a, 'r')) # dict_keys(['images', 'annotations', 'categories'])
train_json_b = json.load(open(train_annotation_path_b, 'r')) # dict_keys(['images', 'annotations', 'categories'])

# ===========================  Test Paths ===============================
test_dir_a = os.path.join(base_dir, r'test\a_images')
test_annotation_path_a = os.path.join(base_dir, r'test\a_annotations.json')
test_dir_b = os.path.join(base_dir, r'test\b_images')
test_annotation_path_b = os.path.join(base_dir, r'test\b_annotations.json')
test_json_a = json.load(open(test_annotation_path_a, 'r')) # dict_keys(['images', 'annotations', 'categories'])
test_json_b = json.load(open(test_annotation_path_b, 'r')) # dict_keys(['images', 'annotations', 'categories'])

# =======================================  preprocess ===============================================
images = train_json_a['images'] # [{'file_name': 'ori_XYGOC20200917193630674-3_0.jpg', 'height': 720, 'width': 960, 'id': 0}]
annotations = train_json_a['annotations'] # {'area': 43560, 'iscrowd': 0, 'image_id': 1190, 'bbox': [551, 255, 198, 220], 'category_id': 19, 'id': 13075}

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
print(results)

with open(r'./train.json', 'w') as f:
	json.dump(results, f)

for img_id, img_infos in results.items():
	img_name = img_infos['img_name']
	img_height = img_infos['img_height']
	img_width = img_infos['img_width']

	img = cv2.imread(os.path.join(train_dir_a, img_name))
	obj = img_infos['obj']
	for o in obj:
		box = o[:4]
		x_min, y_min, w, h = box
		x_max = x_min + w
		y_max = y_min + h

		category = o[4]
		img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255), thickness=2)
		print(id_name_map[category])
		cv2.imshow('img', img)
		cv2.waitKey()
