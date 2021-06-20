import torch
from torch import nn as nn

class ConvReLU(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class Feature_Extrace(nn.Module):
    def __init__(self):
        super(Feature_Extrace, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = ConvReLU(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv12 = ConvReLU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv21 = ConvReLU(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv22 = ConvReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv31 = ConvReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv32 = ConvReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv33 = ConvReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv41 = ConvReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv42 = ConvReLU(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv43 = ConvReLU(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv51 = ConvReLU(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv52 = ConvReLU(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv53 = ConvReLU(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):

        out = self.conv11(x)
        out = self.conv12(out)
        out = self.max_pool(out)

        out = self.conv21(out)
        out = self.conv22(out)
        out = self.max_pool(out)

        out = self.conv31(out)
        out = self.conv32(out)
        out = self.conv33(out)
        out = self.max_pool(out)

        out = self.conv41(out)
        out = self.conv42(out)
        out = self.conv43(out)
        out = self.max_pool(out)

        out = self.conv51(out)
        out = self.conv52(out)
        out = self.conv53(out)

        return out

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.feature_extract = Feature_Extrace()

    def forward(self, x):
        out = self.feature_extract(x)
        return out

