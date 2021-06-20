import torch
import torch.nn as nn
import numpy as np
# import utils
import os
import torch.nn.functional as F
import random
import copy

@torch.no_grad()
def get_spm(input, target, model):

    imgsize = (input.size(2), input.size(3))
    bs = input.size(0)
    with torch.no_grad():
        output,fms = model(input)
        clsw = model.module._fc
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i, target[i].long()])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i,target[i].long()]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps,clslogit

def rand_bbox(size, lam,center=False,attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

@torch.no_grad()
def as_cutmix(input,target,conf,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]

        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = utils.rand_bbox(input.size(), lam)

        if (bby2_1-bby1_1)*(bbx2_1-bbx1_1) > 4 and  (bby2-bby1)*(bbx2-bbx1)>4:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            # adjust lambda to exactly match pixel ratio
            lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a *= torch.ones(input.size(0))
    lam_b = 1 - lam_a

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()

@torch.no_grad()
def cutmix(input,target,conf,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        lam_a *= torch.ones(input.size(0))

    lam_b = 1 - lam_a


    return input,target,target_b,lam_a.cuda(),lam_b.cuda()

@torch.no_grad()
def cutout(input,target,conf=None,model=None):

    r = np.random.rand(1)
    lam = torch.ones(input.size(0)).cuda()
    target_b = target.clone()
    lam_a = lam
    lam_b = 1-lam

    if r < conf.prob:
        bs = input.size(0)
        lam = 0.75
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = 0

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()

@torch.no_grad()
def mixup_data(input, target, beta=0.2, prob=1.0):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    bs = input.size(0)
    target_a = target
    target_b = target

    if r < prob:
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        lam = np.random.beta(beta, beta)
        lam_a = lam_a*lam
        input = input * lam + input[rand_index] * (1-lam)

    lam_b = 1 - lam_a

    mixed_x = input
    return mixed_x, target_a, target_b, lam_a.cuda(), lam_b.cuda()

def mixup_criterion(criterion, pred, target_a, target_b, lam_a, lam_b):
    return lam_a * criterion(pred, target_a) + lam_b * criterion(pred, target_b)

@torch.no_grad()
def snapmix_data(input, target, beta, prob, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target = torch.argmax(target, dim=1)
    target_b = target.clone()

    if r < prob:
        wfmaps,_ = get_spm(input, target, model)
        bs = input.size(0)
        lam = np.random.beta(beta, beta)
        lam1 = np.random.beta(beta, beta)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index,:,:]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

        area = (bby2-bby1)*(bbx2-bbx1)
        area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

        if  area1 > 0 and  area>0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
            lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1-lam

    mixed_x = input
    target_a = target
    return mixed_x, target_a, target_b, lam_a.cuda(), lam_b.cuda()

def snapmix_criterion(criterion, pred, target_a, target_b, lam_a, lam_b):
    return torch.mean(lam_a * criterion(pred, target_a) + lam_b * criterion(pred, target_b))