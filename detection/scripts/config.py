import os.path as osp

base_path = 'detection'

weight_path = osp.join(base_path, 'weights', 'pixlink.h5')
ckpt_path = osp.join(base_path, 'weights', 'conv3_3', 'model.ckpt-38055')

r_mean = 123.
g_mean = 117.
b_mean = 104.
rgb_mean = [r_mean, g_mean, b_mean]
