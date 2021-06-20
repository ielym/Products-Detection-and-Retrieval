import torch
import numpy as np

def area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def cal_batch_iou(box1, box2):
    areas1 = area(box1)
    areas2 = area(box2)

    x_min = torch.max(box1[:, None, 0], box2[:, 0])
    y_min = torch.max(box1[:, None, 1], box2[:, 1])
    x_max = torch.min(box1[:, None, 2], box2[:, 2])
    y_max = torch.min(box1[:, None, 3], box2[:, 3])

    inter = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)

    iou = inter / (areas1[:, None] + areas2 - inter)

    return iou