import cv2
import numpy as np
import os
from glob import glob
import shutil

base_dir = r'/home/ymluo/DataSets/productsDET/V03/train/b_images'
target_dir = r'/home/ymluo/DataSets/productsDET/V04/train/b_images'

for target_category in range(0, 150):
	img_pathes = glob(os.path.join(base_dir, '*-{}.jpg'.format(target_category)))
	if len(img_pathes) == 0:
		continue
	np.random.shuffle(img_pathes)

	# choised_pathes = []
	# for img_path in img_pathes:
	# 	base_name = os.path.basename(img_path)
	# 	if base_name[0] in [str(i) for i in range(10)]:
	# 		print(base_name)

	img_pathes = img_pathes[:136]
	for img_path in img_pathes:
		shutil.copy(img_path, os.path.join(target_dir, os.path.basename(img_path)))
		print(os.path.join(target_dir, os.path.basename(img_path)))