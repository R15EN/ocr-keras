import numpy as np
import os.path as osp
import json
import tensorflow as tf
import matplotlib.pyplot as plt

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

def decode_predictions(pred, max_len, char_set):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0][:, :max_len]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1))).numpy().ravel()
        res = ''.join([char_set[int(i)] for i in res])
        output_text.append(res)
    return output_text

def visualize_predictions(images, predictions):
    _, ax = plt.subplots(4, 4, figsize=(15, 8))
    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        
        title = f"Prediction: {predictions[i]}"
        ax[i // 4, i % 4].imshow(img)
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
    plt.show()