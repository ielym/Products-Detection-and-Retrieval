from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
from glob import glob

if __name__ == '__main__':

	base_dir = r'/home/ymluo/DataSets/productsDET/V04/train/a_images'

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

	catetories_count = np.array(sorted_record).reshape([-1, 2])
	plt.figure(figsize=(20, 5))
	plt.bar(catetories_count[..., 0], catetories_count[..., 1].astype(np.int))
	plt.xticks(rotation=270)
	plt.savefig('./train_a.png')
# plt.show()
