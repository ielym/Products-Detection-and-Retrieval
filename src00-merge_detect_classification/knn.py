import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from albumentations import (
	HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
	Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
	ShiftScaleRotate, CenterCrop, Resize, Rotate
)
from albumentations.pytorch import ToTensorV2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataAugmentation():

	def Train_Transforms(self, img_size):
		return Compose([
			# RandomResizedCrop(img_size[0], img_size[1]),
			Resize(img_size[0], img_size[1], p=1),
			# Transpose(p=1),
			# HorizontalFlip(p=1),
			# VerticalFlip(p=1),
			# ShiftScaleRotate(p=1),
			# HueSaturationValue(hue_shift_limit=(-1, 1), sat_shift_limit=(-1, 1), val_shift_limit=(-1, 1), p=1),
			# RandomBrightnessContrast(brightness_limit=(-1, 1), contrast_limit=(-1, 1), p=1),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
			# CoarseDropout(p=0.5),
			# Cutout(p=0.5, max_h_size=20, max_w_size=20),
			ToTensorV2(p=1.0),
		], p=1.)

class StoreDataset(Dataset):
	def __init__(self, samples, input_shape):
		self.samples = samples

		self.samples_dict = {}

		self.input_shape = input_shape

		self.augmentation = DataAugmentation()
		self.train_transforms = self.augmentation.Train_Transforms(img_size=(self.input_shape[1], self.input_shape[2]))

	def __len__(self):
		return len(self.samples)

	def padding(self, img, target_size):
		ori_height, ori_width, c = img.shape

		new_img = np.zeros([target_size, target_size, c], dtype=img.dtype)
		new_img.fill(128.0)

		scale = target_size / max(ori_height, ori_width)
		scale_height = int(scale * ori_height)
		scale_widht = int(scale * ori_width)
		scale_img = cv2.resize(img, (scale_widht, scale_height))

		new_img[:scale_height, :scale_widht, :] = scale_img

		return new_img


	def preprocess_img(self, img_path):
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img = self.train_transforms(image=img)['image']
		return img

	def preprocess_label(self, label):
		label = torch.tensor(label, dtype=torch.float32)
		return label

	def __getitem__(self, idx):
		anchor_info = self.samples[idx]
		anchor_path = anchor_info[0]
		anchor_category = int(anchor_info[1])

		img_anchor = self.preprocess_img(anchor_path)
		label_anchor = self.preprocess_label(anchor_category)
		return img_anchor, label_anchor

def data_flow_store(store_dir, input_shape):
	store_img_names = os.listdir(store_dir)
	stroe_samples = []
	for store_img_name in store_img_names:
		img_path = os.path.join(store_dir, store_img_name)
		img_category = os.path.basename(store_img_name).split('.')[0].split('-')[1]
		stroe_samples.append((img_path, img_category))
	store_pathes_categories = np.array(stroe_samples).reshape([-1, 2]) # (3965, 2)
	store_dataset = StoreDataset(store_pathes_categories, input_shape)
	return store_dataset

def save_feature(model, store_dir, target_path):
	batch_size = 8
	store_dataset = data_flow_store(store_dir=store_dir, input_shape=(3, 112, 112))
	store_loader = torch.utils.data.DataLoader(store_dataset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True, drop_last=True)

	feature_collect = []
	classes_collect = []
	cnt = 0
	model.eval()
	with torch.no_grad():
		for batch, (images, labels) in enumerate(store_loader):
			images = images.cuda(non_blocking=True)
			labels = labels.cuda(non_blocking=True)
			predict_feature = model(images)
			feature_collect.append(predict_feature)
			classes_collect.append(labels)
			cnt += predict_feature.size(0)
			print('\r Already extract : {} features. '.format(cnt), end='')
		print()
		feature_collect = torch.cat(feature_collect, dim=0)
		classes_collect = torch.cat(classes_collect, dim=0)
		feature_classes = torch.cat([feature_collect, classes_collect.unsqueeze(1)], dim=1)
		feature_classes = feature_classes.cpu().numpy()

		# scaler = StandardScaler()
		# normalization_feature = scaler.fit_transform(feature_classes[..., :-1], feature_classes[..., -1])
		# feature_classes[..., :-1] = normalization_feature

		print('Total Extract Train Features : ', feature_classes.shape)
		np.save(target_path, feature_classes)
		print('Save Feature-class success !')

def get_knn(feature_path):
	feature_classes = np.load(feature_path)
	feature_collect = feature_classes[..., :-1]
	classes_collect = feature_classes[..., -1]

	print('Total Extract Train Features : ', feature_collect.shape, classes_collect.shape)

	# knn = KNeighborsClassifier(n_neighbors=10, p=2, weights='distance', n_jobs=5)
	knn = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=5, metric='cosine')
	knn.fit(feature_collect, classes_collect)
	print('Train KNN Done!')
	return knn
