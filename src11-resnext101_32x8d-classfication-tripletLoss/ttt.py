import numpy as np
import cv2
import pandas as pd
import os
from glob import glob

base_dir = r'E:\kaggle2021-disease'

csv_path = os.path.join(base_dir, 'train.csv')

df = pd.read_csv(csv_path)

categories = df['labels'].unique()

d = {'healthy': 0, 'scab frog_eye_leaf_spot complex': 1, 'scab': 2, 'complex': 3, 'rust': 4, 'frog_eye_leaf_spot': 5, 'powdery_mildew': 6, 'scab frog_eye_leaf_spot': 7, 'frog_eye_leaf_spot complex': 8, 'rust frog_eye_leaf_spot': 9, 'powdery_mildew complex': 10, 'rust complex': 11}

array = df.to_numpy()

for i in range(len(array)):
	array[i, 1] = d[array[i, 1]]

df = pd.DataFrame(data=array, index=None, columns=['image', 'labels'])
df.to_csv('./mytrain.csv')