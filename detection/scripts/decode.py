import numpy as np

def get_neighbours_8(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def is_valid_cord(x, y, w, h):
    return x >= 0 and x < w and y >= 0 and y < h

def decode_image_by_join(pixel_scores, link_scores, 
                         pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)

    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        if update_parent:
            set_parent(point, root)
        return root

    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}

        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(pixel_mask, dtype=np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    for point in points:
        y, x = point
        neighbours = get_neighbours_8(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if is_valid_cord(nx, ny, w, h):
                link_value = link_mask[y, x, n_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:
                    join(point, (ny, nx))

    mask = get_all()

    return mask

def decode_batch(pixel_cls_scores, pixel_link_scores,
                 pixel_conf_threshold=None, link_conf_threshold=None):
    if pixel_conf_threshold is None:
        pixel_conf_threshold = 0.6

    if link_conf_threshold is None:
        link_conf_threshold = 0.9

    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :, 1]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :, 1]
        mask = decode_image_by_join(
            image_pos_pixel_scores, image_pos_link_scores,
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)

