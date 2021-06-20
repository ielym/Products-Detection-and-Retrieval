import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BackBone
from .anchorgenerator import ANchorGenerator


class Swish(nn.Module):
	def __init__(self):
		super(Swish, self).__init__()

	def forward(self, x):
		x = x * F.sigmoid(x)
		return x


class Mish(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self,x):
		x = x * (torch.tanh(F.softplus(x)))
		return x

class YoloBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(YoloBlock, self).__init__()

		self.block = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=in_channels // 2),
			Mish(),

			nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=in_channels),
			Mish(),

			nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=in_channels // 2),
			Mish(),

			nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=in_channels),
			Mish(),

			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=out_channels),
			Mish(),
		)

	def forward(self, x):
		return self.block(x)

class DetectBottle(nn.Module):
	def __init__(self, large_channels, middle_channels, small_channels):
		super(DetectBottle, self).__init__()

		self.yolo_block1 = YoloBlock(small_channels, 512)
		self.squeeze1 = nn.Sequential(
			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=256),
			Mish(),
		)

		self.yolo_block2 = YoloBlock(middle_channels+256, 256)
		self.squeeze2 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=128),
			Mish(),
		)

		self.yolo_block3 = YoloBlock(large_channels+128, 128)


		self._initial_layers()
	def _initial_layers(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x_large, x_middle, x_small):

		feature_small = self.yolo_block1(x_small) # torch.Size([2, 512, 8, 8])

		feature_cat1 = self.squeeze1(feature_small) # torch.Size([2, 256, 8, 8])
		up_feature_cat1 = F.interpolate(feature_cat1, size=(feature_cat1.size(2)*2, feature_cat1.size(3)*2), mode='bilinear', align_corners=True) # torch.Size([2, 256, 16, 16])
		feature_middle = self.yolo_block2(torch.cat([x_middle, up_feature_cat1], dim=1)) # torch.Size([2, 256, 16, 16])

		feature_cat2 = self.squeeze2(feature_middle) # torch.Size([2, 128, 16, 16])
		up_feature_cat2 = F.interpolate(feature_cat2, size=(feature_cat2.size(2)*2, feature_cat2.size(3)*2), mode='bilinear', align_corners=True) # torch.Size([2, 128, 32, 32])
		feature_large = self.yolo_block3(torch.cat([x_large, up_feature_cat2], dim=1)) # torch.Size([2, 128, 32, 32])

		return feature_large, feature_middle, feature_small

class Predictor(nn.Module):

	def __init__(self, in_channels, num_anchors_per_level, num_classes):
		super(Predictor, self).__init__()

		self.large_predictor = nn.Sequential(
			nn.Conv2d(in_channels[0], in_channels[0]*2, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=in_channels[0]*2),
			Mish(),
			nn.Conv2d(in_channels[0]*2, num_anchors_per_level[0]*(5+num_classes), 1, stride=1, padding=0, bias=True)
		)

		self.middle_predictor = nn.Sequential(
			nn.Conv2d(in_channels[1], in_channels[1]*2, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=in_channels[1]*2),
			Mish(),

			nn.Conv2d(in_channels[1]*2, num_anchors_per_level[1]*(5+num_classes), 1, stride=1, padding=0, bias=True)
		)

		self.small_predictor = nn.Sequential(
			nn.Conv2d(in_channels[2], in_channels[2]*2, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=in_channels[2]*2),
			Mish(),

			nn.Conv2d(in_channels[2]*2, num_anchors_per_level[2]*(5+num_classes), 1, stride=1, padding=0, bias=True)
		)

		self._initial_layers()

	def _initial_layers(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, feature_large, feature_middle, feature_small):
		predict_large = self.large_predictor(feature_large) # torch.Size([2, 75, 32, 32])
		predict_middle = self.middle_predictor(feature_middle) # torch.Size([2, 75, 16, 16])
		predict_small = self.small_predictor(feature_small) # torch.Size([2, 75, 8, 8])
		return predict_large, predict_middle, predict_small


class Yolo3(nn.Module):
	def __init__(self,
				 backbone_name, backbone_weights, backbone_pretrained,

				 yolo_params,
				 ):
		super(Yolo3, self).__init__()

		self.mode = yolo_params['mode']
		self.num_classes = yolo_params['num_classes']
		self.num_anchors_per_level = yolo_params['num_anchors_per_level']
		self.bg_threshold = yolo_params['bg_threshold']
		self.data_local = yolo_params['data_local']

		self.prob_threshold = yolo_params['prob_threshold']

		if self.mode == 'Train':
			self.anchors = ANchorGenerator(self.data_local, k=sum(self.num_anchors_per_level), iters=200, mode=self.mode).get_anchors()
		else:
			self.anchors = torch.tensor([[0.05007908, 0.09432577],
										 [0.14074688, 0.15466505],
										 [0.13035001, 0.35630974],
										 [0.34889045, 0.28340921],
										 [0.22714622, 0.56508315],
										 [0.38271534, 0.67783111],
										 [0.73001283, 0.43364769],
										 [0.59752166, 0.80314022],
										 [0.89895731, 0.85308385]], dtype=torch.float).view(-1, 2)

		self.backbone = BackBone(backbone_name=backbone_name, backbone_weights=backbone_weights, pretrained=backbone_pretrained)

		large_channels, middle_channels, small_channels = self.backbone.backbone_feature_channels()
		self.detect_bottle = DetectBottle(large_channels, middle_channels, small_channels)

		self.predictor = Predictor(in_channels=[128, 256, 512], num_anchors_per_level=self.num_anchors_per_level, num_classes=self.num_classes)

	def forward(self, x, targets=None, iteration=None):

		x_large, x_middle, x_small = self.backbone(x) # torch.Size([2, 512, 32, 32]) torch.Size([2, 1024, 16, 16]) torch.Size([2, 2048, 8, 8])
		feature_large, feature_middle, feature_small = self.detect_bottle(x_large, x_middle, x_small) # torch.Size([2, 128, 32, 32]) torch.Size([2, 256, 16, 16]) torch.Size([2, 512, 8, 8])
		predict_large, predict_middle, predict_small = self.predictor(feature_large, feature_middle, feature_small) # torch.Size([2, 75, 32, 32]) torch.Size([2, 75, 16, 16]) torch.Size([2, 75, 8, 8])

		input_size = x.size(2)
		device = x.device

		anchor_levels = []
		anchors = self.anchors.reshape([len(self.num_anchors_per_level), -1, 2]).to(x)
		for idx in range(len(self.num_anchors_per_level)):
			anchor_levels.append(anchors[idx, :, :])

		predict_levels = [predict_large, predict_middle, predict_small]

		feature_size_levels = [predict_large.size(2), predict_middle.size(2), predict_small.size(2)]

		results = []
		loss = {}
		if self.mode == 'Train':
			from .loss_yolo3 import Loss_Yolo
			yolo_loss = Loss_Yolo(
								predict_levels=predict_levels,
								targets=targets,
								anchor_levels=anchor_levels,
								num_anchor_levels=self.num_anchors_per_level,
								feature_size_levels=feature_size_levels,
								bg_threshold=self.bg_threshold,
								num_classes=self.num_classes,
								device=device,
								iteration=iteration,
						 )
			loss['Loss_Yolo'] = yolo_loss
		else:
			from .loss_yolo3 import post_process
			results = post_process(
								predict_levels=predict_levels,
								anchor_levels=anchor_levels,
								num_anchor_levels=self.num_anchors_per_level,
								feature_size_levels=feature_size_levels,
								input_size=input_size,
								num_classes=self.num_classes,
								device=device,
								prob_threshold=self.prob_threshold,
			)
		return results, loss