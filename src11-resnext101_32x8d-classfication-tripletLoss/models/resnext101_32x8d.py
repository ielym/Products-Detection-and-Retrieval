import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.lym_fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.lym_fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = self.lym_fc2(self.relu1(self.lym_fc1(self.avg_pool(x))))
		max_out = self.lym_fc2(self.relu1(self.lym_fc1(self.max_pool(x))))
		out = avg_out + max_out
		out = self.sigmoid(out) * x
		return out

class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()

		assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1

		self.lym_conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		out = torch.cat([avg_out, max_out], dim=1)
		out = self.lym_conv1(out)
		out = self.sigmoid(out) * x
		return out

class Self_Attn(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize, C, width,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        # print proj_query.size()
        # print proj_key.size()
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.lym_self_atten_head = Self_Attn(in_dim=self.inplanes)
        # self.lym_se_head = ChannelAttention(in_planes=self.inplanes, ratio=16)
        # self.lym_sa_head = SpatialAttention(kernel_size=7)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # self.lym_se_tail = ChannelAttention(in_planes=self.inplanes, ratio=16)
        # self.lym_sa_tail = SpatialAttention(kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self._fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature_fc = nn.Linear(512 * block.expansion, 512)
        self.feature_bn = nn.BatchNorm1d(num_features=512)

        self._fc = nn.Linear(512, num_classes)

        # self.fc1 = nn.Linear(512 * block.expansion, 1024)
        # self.lym_dropout = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(1024, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.lym_se_head(x)
        # x = self.lym_sa_head(x)

        x = self.maxpool(x)

        # x = self.lym_self_atten_head(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4 = self.layer4(x)

        # x = self.lym_se_tail(x)
        # x = self.lym_sa_tail(x)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)

        feature = self.feature_fc(x)
        # feature = self.feature_bn(x)

        predict = self._fc(feature)
        # feature = self.feature_bn(x)
        # feature = self.feature_fc(x)
        # mean_x = torch.mean(x, dim=0)
        # var_x = torch.var(x, dim=0)
        # x = (x - mean_x) / torch.sqrt(var_x)
        return predict, feature

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)