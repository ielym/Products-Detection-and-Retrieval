import cv2
import numpy as np

def HorizontalFlip(img, box, p):
	alpha = np.random.uniform(0, 1)
	if alpha > p:
		return img, box

	img = cv2.flip(img, flipCode=1)
	h, w, c = img.shape
	box[:, 2], box[:, 0] = w - box[:, 0],  w - box[:, 2]
	return img, box

def VerticalFlip(img, box, p):
	alpha = np.random.uniform(0, 1)
	if alpha > p:
		return img, box

	img = cv2.flip(img, flipCode=0)
	h, w, c = img.shape
	box[:, 3], box[:, 1] = h - box[:, 1],  h - box[:, 3]
	return img, box