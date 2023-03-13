import numpy as np
import cv2

def rect_to_xys(rect, image_shape):
    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points


def min_area_rect(cnt):
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h


def mask_to_bboxes(mask, image_shape=None, min_area=None, min_height=None):
    image_h, image_w = image_shape[0:2]

    if min_area is None:
        min_area = 300

    if min_height is None:
        min_height = 10
        
    bboxes = []
    max_bbox_idx = mask.max()
    mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
        cnts, _ = cv2.findContours(bbox_mask.astype(np.uint8), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue

        if rect_area < min_area:
            continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)

    return bboxes

def resize_image(image):
    height = image.shape[0]
    width = image.shape[1]
    
    ratio_h = np.ceil(height / 32)
    ratio_w = np.ceil(width / 32)
    
    image = cv2.resize(image, (int(ratio_w * 32), int(ratio_h * 32))) 
    return image