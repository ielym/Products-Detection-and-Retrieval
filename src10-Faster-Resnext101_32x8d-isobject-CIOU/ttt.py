import json
import os
from glob import glob
import cv2
import numpy

json_dict = json.load(open('./predictions.json', 'r'))

images = json_dict['images']
annotations = json_dict['annotations']

print(images)
# print(annotations)

img_name_id_map = {}
for line in images:
	pass