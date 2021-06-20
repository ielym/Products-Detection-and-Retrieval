import numpy as np
import torch

def transform_delta_2_real(pre_deltas, anchors):
    dx = pre_deltas[:, 0]
    dy = pre_deltas[:, 1]
    dw = pre_deltas[:, 2]
    dh = pre_deltas[:, 3]

    x_mins = anchors[:, 0]
    y_mins = anchors[:, 1]

    Pw = anchors[:, 2] - anchors[:, 0] + 1
    Ph = anchors[:, 3] - anchors[:, 1] + 1
    Px = x_mins + (Pw-1) * 0.5
    Py = y_mins + (Ph-1) * 0.5

    Gw = Pw * np.exp(dw)
    Gh = Ph * np.exp(dh)
    Gx = Px + Pw * dx
    Gy = Py + Ph * dy

    pre_boxes = np.zeros([pre_deltas.shape[0], pre_deltas.shape[1]], dtype=np.int)
    pre_boxes[:, 0] = Gx - (Gw-1) * 0.5
    pre_boxes[:, 1] = Gy - (Gh-1) * 0.5
    pre_boxes[:, 2] = Gx + (Gw-1) * 0.5
    pre_boxes[:, 3] = Gy + (Gh-1) * 0.5

    return pre_boxes

def transform_real_2_delta(gt_boxes, anchors):
    Pw = anchors[:, 2] - anchors[:, 0] + 1
    Ph = anchors[:, 3] - anchors[:, 1] + 1
    Px = anchors[:, 0] + (Pw - 1) * 0.5
    Py = anchors[:, 1] + (Ph - 1) * 0.5

    Gw = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    Gh = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    Gx = gt_boxes[:, 0] + (Gw - 1) * 0.5
    Gy = gt_boxes[:, 1] + (Gh - 1) * 0.5

    gt_deltas = np.zeros([anchors.shape[0], anchors.shape[1]], dtype=np.float)
    gt_deltas[:, 0] = (Gx - Px) / Pw
    gt_deltas[:, 1] = (Gy - Py) / Ph
    gt_deltas[:, 2] = np.log(Gw / Pw)
    gt_deltas[:, 3] = np.log(Gh / Ph)

    return gt_deltas