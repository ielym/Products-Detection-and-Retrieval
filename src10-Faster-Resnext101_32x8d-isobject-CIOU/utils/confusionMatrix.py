# -*- coding: utf-8 -*-
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import os
import random

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']  #可显示中文字符
plt.rcParams['axes.unicode_minus']=False

def draw_confusion_matrix(num_classes, pre_categories, true_categories, classes):

	confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
	for gt_category, pre_categorie in zip(true_categories, pre_categories):
		confusion_matrix[int(gt_category), int(pre_categorie)] += 1
	plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
	plt.title('Confusion Matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=-45)
	plt.yticks(tick_marks, classes)

	# ij配对，遍历矩阵迭代器
	iters = np.reshape([[[i, j] for j in range(num_classes)] for i in range(num_classes)], (confusion_matrix.size, 2))
	for i, j in iters:
		plt.text(j, i, format(confusion_matrix[i, j]), fontsize=7)  # 显示对应的数字

	plt.ylabel('True')
	plt.xlabel('Predict')
	plt.tight_layout()
	plt.savefig('./result.png')
	plt.show()
	return confusion_matrix
