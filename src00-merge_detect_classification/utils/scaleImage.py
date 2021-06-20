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

def scale_img_minsize(ori_h, ori_w, minsize=600):
	if ori_w > ori_h:
		target_h = minsize
		target_w = 1.0 * ori_w * target_h / ori_h
	else:
		target_w = minsize
		target_h = 1.0 * ori_h * target_w / ori_w

	target_h = int(target_h)
	target_w = int(target_w)

	scale_h = 1.0 * target_h / ori_h
	scale_w = 1.0 * target_w / ori_w

	return target_h, target_w, scale_h, scale_w

def scale_img_maxsize(ori_h, ori_w, maxsize=600):
	if ori_w > ori_h:
		target_w = maxsize
		target_h = 1.0 * ori_h * target_w / ori_w
	else:
		target_h = maxsize
		target_w = 1.0 * ori_w * target_h / ori_h

	target_h = int(target_h)
	target_w = int(target_w)

	scale_h = 1.0 * target_h / ori_h
	scale_w = 1.0 * target_w / ori_w

	return target_h, target_w, scale_h, scale_w
