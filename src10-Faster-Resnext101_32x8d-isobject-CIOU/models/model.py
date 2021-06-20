import torchvision
import torch
import torch.nn as nn
import collections
import cv2
import numpy as np
from .FasterRcnn import FasterRCNN

def _fasterrcnn(**kwargs):
	model = FasterRCNN(**kwargs)
	return model

def FasterRcnn(**kwargs):
	model = _fasterrcnn(**kwargs)
	return model