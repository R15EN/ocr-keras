import os.path as osp

base_path = 'recognition'
data_path = osp.join(base_path, 'data')
path_to_images = osp.join(data_path, 'images')
path_to_labels = osp.join(data_path, 'gt.txt')
path_to_model = osp.join(base_path, 'trained_model', 'ocr_model')

image_size = (256, 64)
epochs = 50
batch_size = 128
