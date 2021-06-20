import numpy as np

def cal_single_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    x_min_c = max(x_min1, x_min2)
    y_min_c = max(y_min1, y_min2)
    x_max_c = min(x_max1, x_max2)
    y_max_c = min(y_max1, y_max2)

    if x_max_c<=x_min_c or y_max_c<=y_min_c:
        return 0

    S_c = (y_max_c - y_min_c) * (x_max_c - x_min_c)

    S1 = (y_max1 - y_min1) * (x_max1 - x_min1)
    S2 = (y_max2 - y_min2) * (x_max2 - x_min2)

    iou = S_c / (S1 + S2 - S_c)

    return iou

def cal_batch_iou(box1, box2):

    x_min_c = np.maximum(box1[:, 0], box2[:, 0])
    y_min_c = np.maximum(box1[:, 1], box2[:, 1])
    x_max_c = np.minimum(box1[:, 2], box2[:, 2])
    y_max_c = np.minimum(box1[:, 3], box2[:, 3])

    S_c = np.maximum(y_max_c - y_min_c, 0) * np.maximum(x_max_c - x_min_c, 0)

    S1 = (box1[:, 3] - box1[:, 1]) * (box1[:, 2] - box1[:, 0])
    S2 = (box2[:, 3] - box2[:, 1]) * (box2[:, 2] - box2[:, 0])

    iou = S_c / (S1 + S2 - S_c)
    return iou