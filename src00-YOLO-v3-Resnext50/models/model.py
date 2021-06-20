import torchvision
import torch
import torch.nn as nn
import collections
import cv2
import numpy as np
from .YOLO import Yolo3

def _yolo(**kwargs):
	model = Yolo3(**kwargs)
	return model

def YOLO(**kwargs):
	model = _yolo(**kwargs)
	return model