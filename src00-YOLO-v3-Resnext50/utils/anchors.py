import numpy as np

def get_base_anchor(feature_stride):
    x_min = 0
    y_min = 0
    x_max = feature_stride - 1
    y_max = feature_stride - 1

    return np.array([x_min, y_min, x_max, y_max], dtype=np.int)

def get_center(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_center = anchor[0] + 0.5 * (w - 1)
    y_center = anchor[1] + 0.5 * (h - 1)
    return w, h, x_center, y_center

def make_anchor(w, h, x_center, y_center):

    x_min = x_center - 0.5 * (w - 1)
    y_min = y_center - 0.5 * (h - 1)
    x_max = x_min + w - 1
    y_max = y_min + h - 1

    return [x_min, y_min, x_max, y_max]

def get_ratios(ratios, base_anchor):
    w, h, x_center, y_center = get_center(base_anchor)
    S = w * h

    ratio_anchors = []
    for ratio in ratios:
        new_h = np.round(np.sqrt(S/ratio))
        new_w = ratio * new_h
        ratio_anchors.append(make_anchor(new_w, new_h, x_center, y_center))
    ratio_anchors = np.array(ratio_anchors, dtype=np.int).reshape([-1, 4])
    return ratio_anchors

def get_scales(scales, ratio_anchors):
    scale_anchors = []
    for scale in scales:
        for anchor in ratio_anchors:
            w, h, x_center, y_center = get_center(anchor)
            new_w = w * scale
            new_h = h * scale
            scale_anchors.append(make_anchor(new_w, new_h, x_center, y_center))
    scale_anchors = np.array(scale_anchors, dtype=np.int).reshape([-1, 4])
    return scale_anchors

def get_k_anchors(feature_stride, ratios, scales):
    base_anchor = get_base_anchor(feature_stride)

    ratio_anchors = get_ratios(ratios, base_anchor)
    scale_anchors = get_scales(scales, ratio_anchors)

    return scale_anchors

def generate_anchors(feature_stride, ratios, scales, feature_width, feature_height):
    k_anchors = get_k_anchors(feature_stride=feature_stride, ratios=ratios, scales=scales)

    shift_x = np.arange(0, feature_width) * feature_stride
    shift_y = np.arange(0, feature_height) * feature_stride
    all_anchors = []
    for x in shift_x:
        for y in shift_y:
            for anchor in k_anchors:
                all_anchors.append(anchor + [x, y, x, y])
    all_anchors = np.array(all_anchors, dtype=np.int)
    all_anchors = all_anchors.reshape([-1, 4])

    return all_anchors

if __name__ == '__main__':
    anchors = generate_anchors(feature_stride=16, ratios=[0.5, 1, 2], scales=[8, 16, 32])
    print(anchors)
    anchors += 500
    import cv2
    img = np.ones(shape=[1000, 1000, 3])
    img.fill(255.)
    for anchor in anchors:
        cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), color=(0, 0, 255), thickness=1)
    cv2.imshow('img', img)
    cv2.waitKey()