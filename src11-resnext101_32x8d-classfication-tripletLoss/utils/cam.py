import torch
import torch.nn.functional as F
import numpy as np
import cv2
from .drawheatmap import drawheatmap

def CAM(input, target, model, ori_img, threshold=30):
    '''
        input : A single image with torch format and size like torch.Size([1, 3, 224, 224]),
        		and it must be the same pre_processing like train or test
        target : The target of the input, also with torch format and size like torch.Size([1]) (argmax(one_hot))

        output, fms = model(input)
        	e.g., resnext101
        		x4 = self.layer4(x)
				x = self.avgpool(x4)
				x = torch.flatten(x, 1)
				x = self._fc(x)
				return x, x4

		ori_img : the ori image, it's Opencv format with shape like (224, 224, 3), and pixel level between [0, 255

		threshold : active score threshold for heatmap

		:return Opencv image with shape lise (224, 224, 3), and pixel level between [0, 255]
        '''
    imgsize = (input.size(2), input.size(3))
    ori_img = ori_img / 255.

    model.eval()
    with torch.no_grad():
        output,fms = model(input)
        clsw = model._fc
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)
        out = F.conv2d(fms, weight, bias=bias)

        outmaps = out[0,target[0].long()]
        outmaps = torch.unsqueeze(outmaps, dim=0)

        outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
        outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)
        outmaps = torch.squeeze(outmaps, dim=0)

        hot_img = outmaps.cpu().numpy()
        hot_img = np.transpose(hot_img, [1,2,0])
        hot_img = np.clip(hot_img, 0, 255) / np.max(hot_img)
        heat_map = drawheatmap(ori_img, hot_img, threshold)

    return heat_map

# if __name__ == '__main__':
#     import cv2
#     import torch
#     from albumentations import (
#         HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
#         Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#         IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
#         IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
#         ShiftScaleRotate, CenterCrop, Resize
#     )
#     from albumentations.pytorch import ToTensorV2
#
#     from models.model import ResNet101
#     from drawheatmap import drawheatmap
#
#     img_size = (448, 448)
#
#     trans = Val_Transforms(img_size)
#
#     ori_img = cv2.imread(r'S:\DataSets\cassava-leaf-disease-classification\train_images\1107438.jpg')
#     img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
#     img = trans(image=img)['image']
#     input = torch.unsqueeze(img, dim=0)
#     target = torch.tensor([3], dtype=torch.float32)
#     input = input.cuda()
#     target = target.cuda()
#
#     model = ResNet101(weights=None, input_shape=(3, img_size[0], img_size[1]), num_classes=5)
#     pretrained_dict = torch.load(r'./models/ep00056-val_acc@1_88.5460-val_lossFocalCosine_0.1470.pth')
#     single_dict = {}
#     for k, v in pretrained_dict.items():
#         single_dict[k[7:]] = v
#     model.load_state_dict(single_dict, strict=True)
#     model = model.cuda()
#
#     heat_map = CAM(input, target, model, cv2.resize(ori_img, img_size) / 255., threshold=30)
#     cv2.imshow('heat_map', heat_map)
#     cv2.waitKey()