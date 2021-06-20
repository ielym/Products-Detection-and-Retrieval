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

def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
	if (isinstance(img, np.ndarray)):
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(img)
	fontStyle = ImageFont.truetype(
		"font/simsun.ttc", textSize, encoding="utf-8")
	draw.text((left, top), text, textColor, font=fontStyle)
	return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def showTorchImg(img):
	img = img.numpy()
	img = np.transpose(img, [1,2,0])
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.imshow('img', img)
	cv2.waitKey()
