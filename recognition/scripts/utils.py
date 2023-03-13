import numpy as np
import os.path as osp
import json

def load_data_from_file(path, shuffle=True):
    words_list = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            words_list.append(tuple(line))
    if shuffle: 
        np.random.shuffle(words_list)
    return words_list

def split_data(words, test_size=0.1):
    split_index = int((1-test_size) * len(words))
    train = words[:split_index]
    test = words[split_index:]
    return train, test

def get_paths_and_labels(samples):
    return list(map(list, list(zip(*samples))))

def save_dataset_config(path, data):
    path = osp.join(path, 'dataset_config.json')
    with open(path, 'w') as f:
        json.dump(data, f)

def load_dataset_config(path):
    path = osp.join(path, 'dataset_config.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data