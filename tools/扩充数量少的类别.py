import numpy as np
import cv2
import os
from glob import glob

base_dir = r'/home/ymluo/DataSets/productsDET/V03/train/a_images'

class Transform():

	def save_img(self, trans_img, img_path, pre_fix):
		base_dir = os.path.dirname(img_path)
		base_name = os.path.basename(img_path)
		trans_name = pre_fix + base_name
		trans_path = os.path.join(base_dir, trans_name)
		cv2.imwrite(trans_path, trans_img)
		# cv2.imshow('trans_img', trans_img)
		# cv2.waitKey()
		print(trans_path)

	def HFlip(self, img, img_path, pre_fix):
		trans_img = cv2.flip(img, flipCode=1)

		self.save_img(trans_img, img_path, pre_fix)


	def VFlip(self, img, img_path, pre_fix):
		trans_img = cv2.flip(img, flipCode=0)

		self.save_img(trans_img, img_path, pre_fix)

	def Transpose(self, img, img_path, pre_fix):
		trans_img = cv2.transpose(img)

		self.save_img(trans_img, img_path, pre_fix)

	def Rotation(self, img, img_path, ration, pre_fix):
		h, w = img.shape[:2]
		M = cv2.getRotationMatrix2D((w//2, h//2), ration, 1.0)
		trans_img = cv2.warpAffine(img, M, (w, h))

		self.save_img(trans_img, img_path, pre_fix)

	def Affine(self, img, img_path, pre_fix):
		h, w = img.shape[:2]
		point1 = np.float32([[50, 50], [300, 50], [50, 200]])
		point2 = np.float32([[10, 50], [300, 50], [100, 250]])
		M = cv2.getAffineTransform(point1, point2)
		trans_img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

		self.save_img(trans_img, img_path, pre_fix)

	def Move(self, img, img_path, pre_fix):
		h, w = img.shape[:2]
		move_h = int(h * 0.1)
		move_w = int(w * 0.1)
		M = np.float32([[1, 0, move_w], [0, 1, move_h]])
		trans_img = cv2.warpAffine(img, M, (w, h))

		self.save_img(trans_img, img_path, pre_fix)

	def Channel_shuffle(self, img, img_path, channels, pre_fix):
		b = img[..., 0:1]
		g = img[..., 1:2]
		r = img[..., 2:3]
		bgr = (b, g, r)
		trans_list = [bgr[i] for i in channels]
		trans_img = np.concatenate(trans_list, axis=2)

		self.save_img(trans_img, img_path, pre_fix)

	def Noise(self, img, img_path, mean, var, pre_fix):
		img = np.array(img / 255, dtype=float)
		noise = np.random.normal(mean, var ** 0.5, img.shape)
		trans_img = img + noise

		self.save_img(trans_img, img_path, pre_fix)


trans = Transform()

target_category = 45
img_pathes = glob(os.path.join(base_dir, '*-{}.jpg'.format(target_category)))

for img_path in img_pathes:

	ori_img = cv2.imread(img_path)

	trans.HFlip(ori_img, img_path, 'H')
	trans.VFlip(ori_img, img_path, 'V')
	trans.Transpose(ori_img, img_path, 'T')
	trans.Move(ori_img, img_path, 'M')

	trans.Affine(ori_img, img_path, 'A')

	trans.Noise(ori_img, img_path, 0, 0.01, 'N1')
	trans.Noise(ori_img, img_path, 0, 0.02, 'N2')

	trans.Channel_shuffle(ori_img, img_path, (0, 2, 1), 'C1')
	trans.Channel_shuffle(ori_img, img_path, (1, 0, 2), 'C2')
	trans.Channel_shuffle(ori_img, img_path, (1, 2, 0), 'C3')
	trans.Channel_shuffle(ori_img, img_path, (2, 0, 1), 'C4')
	trans.Channel_shuffle(ori_img, img_path, (2, 1, 0), 'C5')

	trans.Rotation(ori_img, img_path, 35, 'R1')
	trans.Rotation(ori_img, img_path, -35, 'R2')
	trans.Rotation(ori_img, img_path, 75, 'R3')
	trans.Rotation(ori_img, img_path, -75, 'R4')

img_names = os.listdir(base_dir)
records = {}
for img_name in img_names:
	category = int(img_name.split('.')[0].split('-')[1])
	if not category in records.keys():
		records[category] = 1
	else:
		records[category] += 1
sorted_record = sorted(records.items(), key=lambda x : x[1])
print(sorted_record)
print(records[target_category])

