import os
import time
from glob import glob
import numpy as np
import cv2
import torch
import json

from utils.showimg import cv2ImgAddText
from classfication import Classfication
from detection import Detection
from object_cfg import args
from knn import save_feature, get_knn


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

classfication_weights = r'./model_classfication/ep00018-val_Loss_Classification_2.6093-val_Loss_Distance_0.0000.pth'
classfication = Classfication(model_weights=classfication_weights,num_classes=116,input_height=112,input_width=112)

# args.inference_weights = r'./model_detection/ep00192-val_rpn_clss_Loss_0.0012-val_rpn_bbox_Loss_0.0009--val_fast_clss_Loss_0.0320-val_fast_bbox_Loss_0.0141.pth'
args.inference_weights = r'./model_detection/ep00651-val_rpn_clss_Loss_0.0005-val_rpn_bbox_Loss_0.0010--val_fast_clss_Loss_0.0504-val_fast_bbox_Loss_0.0148.pth'

detection = Detection(args)

# ======================================================================================================================
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


classfication_model = classfication.get_model()
detection_model = detection.get_model()
classfication_model.eval()
detection_model.eval()

store_dir = r'S:\DataSets\productsDET\V02\test\b_images'
feature_path = r'./extrace_feature-Resize.npy'
# save_feature(classfication_model, store_dir, feature_path)
# feature_path = r'./merge.npy'
# feature_path = r'./balance.npy'

is_show = True

knn = get_knn(feature_path)

TEST_DIR = r'S:\DataSets\productsDET\V01\test\a_images'
test_images = glob(os.path.join(TEST_DIR, '*'))

# np.random.shuffle(test_images)

json_DIR = os.path.join(r'S:\DataSets\productsDET\V01\test', 'a_annotations.json')
annotations_json = json.load(open(json_DIR, 'r'))
annotations_images = annotations_json['images']
img_name_id_map = {}
img_name_size_map = {}
for image_info in annotations_images:
	img_name_id_map[image_info['file_name']] = image_info['id']
	img_name_size_map[image_info['file_name']] = (int(image_info['height']), int(image_info['width']))

class FindUnion():
	def __init__(self, n):
		self.n = n
		self.record = list(range(n))

	def find(self, v):
		return self.record[v]

	def union(self, v1, v2):
		root1 = self.find(v1)
		root2 = self.find(v2)
		if root1 == root2:
			return
		for i in range(self.n):
			if self.record[i] == root2:
				self.record[i] = root1

	def issame(self, v1, v2):
		return self.find(v1) == self.find(v2)

images = []
annotations = []
with torch.no_grad():
	start = time.time()
	for cnt, img_path in enumerate(test_images):
		print('\r {} / {}'.format(cnt + 1, len(test_images)), end='')

		ori_img = cv2.imread(img_path)
		rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

		boxes, probs = detection.predict(detection_model, rgb_img) # (31, 4) (31,)
		boxes = boxes.round().astype(np.int)

		file_name = os.path.basename(img_path)
		img_id = img_name_id_map[file_name]
		images.append({"file_name": file_name, "id": img_id})

		all_images = []
		for box in boxes:
			x_min, y_min, x_max, y_max = box
			crop_img = rgb_img[y_min:y_max+1, x_min:x_max+1]
			all_images.append(crop_img)
		all_features = classfication.predict(classfication_model, all_images) # (24, 512)

		category_ids = knn.predict(all_features) # (31,)

		# ==============================================================================================================

		top = np.sum(all_features[:, None] * all_features, axis=2) # (41, 41)
		sss = np.sqrt(np.sum(np.square(all_features), axis=1)) # (44,)
		bottom = sss[:, None] * sss # (41, 41)
		cosine_distance =  top / bottom

		n = cosine_distance.shape[0]
		uf = FindUnion(n)
		for i in range(n):
			for j in range(n):
				if cosine_distance[i, j] >= 0.7:
					if not uf.issame(i, j):
						uf.union(i, j)
		merge = {}
		for i in range(n):
			root = uf.find(i)
			if not root in merge.keys():
				merge[root] = [[i], [category_ids[i]]]
			else:
				merge[root][0].append(i)
				merge[root][1].append(category_ids[i])
		for k, v in merge.items():
			idxs = v[0]
			categories = v[1]
			counts = np.bincount(np.array(categories, dtype=np.int))
			vote_category = np.argmax(counts)
			for i in idxs:
				category_ids[i] = vote_category



		# ==============================================================================================================

		if is_show:
			show_img = ori_img.copy()
		for box, prob, categroy_id in zip(boxes, probs, category_ids):
			x_min, y_min, x_max, y_max = box
			w = x_max - x_min
			h = y_max - y_min
			single_dict = {
				"image_id": int(img_id),
				"category_id": int(categroy_id),
				"bbox": [float(x_min), float(y_min), float(w), float(h)],
				"score": float(prob)
			}
			annotations.append(single_dict)

			if is_show:
				show_img = cv2.rectangle(show_img, (x_min, y_min), (x_max, y_max), color=colors[categroy_id], thickness=2)
				show_img = cv2.rectangle(show_img, (x_min, y_min), (int(x_min + 128), int(y_min - 32)), color=colors[categroy_id], thickness=-1)
				show_img = cv2ImgAddText(show_img, '\t' + str(id_name_map[categroy_id]), x_min - 24, y_min - 24, textColor=(255, 255, 255), textSize=24)
		if is_show:
			cv2.imshow('img', show_img)
			cv2.waitKey()

print('\n', (time.time() - start) / len(test_images), ' sec per image avg.')

predictions = {"images": images, "annotations": annotations}
result_json = json.dumps(predictions)
with open("predictions.json", "w") as f:
	f.write(result_json)
