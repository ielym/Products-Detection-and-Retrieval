#coding:utf-8
import torch
import torch.nn as nn
import math
import torch.functional as F
from collections import OrderedDict

from .anchorgenerator import ANchorGenerator
from .backbone import BackBone
from .rpn import RegionProposalNetwork
from .fastrcnn import FastHead

class FasterRCNN(nn.Module):

	def __init__(self, args):
		super(FasterRCNN, self).__init__()

		self.mode = args.mode

		self.backbone = BackBone(backbone_name=args.backbone_name, backbone_weights=args.backbone_weights, pretrained=args.backbone_pretrained)
		backbone_out_channels = self.backbone.backbone_feature_channels()
		backbone_last_channels = self.backbone.backbone_last_channels()

		if self.mode == 'Train':
			self.anchor_wh = ANchorGenerator(args.data_local, k=args.num_anchor, iters=200).get_anchors()
			print(self.anchor_wh)
		else:
			self.anchor_wh = torch.tensor([[0.06852354, 0.08944999],
											[0.06713755, 0.13125266],
											[0.12136354, 0.0748318 ],
											[0.08923748, 0.11355648],
											[0.10842337, 0.14676274],
											[0.19183247, 0.12691902],
											[0.13736416, 0.19570282],
											[0.195977,   0.25839058],
											[0.3042669,  0.38802502]], dtype=torch.float).view(-1, 2)

		self.rpn = RegionProposalNetwork(backbone_channels=backbone_out_channels, args=args)

		self.fast = FastHead(args, self.backbone.feature_extract.layer4, backbone_last_channels)

	def forward(self, images, targets=None):
		input_size = (images.size(2), images.size(3))

		backbone_feature = self.backbone(images) # torch.Size([4, 1024, 32, 32])

		feature_size = (backbone_feature.size(2), backbone_feature.size(3))

		proposals, loss_rpn = self.rpn(backbone_feature, self.anchor_wh.clone(), input_size, feature_size, targets)

		predicted, loss_fast = self.fast(backbone_feature, proposals, input_size, feature_size, targets)

		return predicted, loss_rpn, loss_fast
