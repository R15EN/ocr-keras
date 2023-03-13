import recognition.scripts.config as config
import recognition.scripts.utils as utils
from recognition.scripts.dataset import OCRDataset

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def decode_batch_predictions(pred, max_len, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def visualize_predictions(images, predictions):
    _, ax = plt.subplots(4, 4, figsize=(15, 8))
    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]
        
        title = f"Prediction: {predictions[i]}"
        ax[i // 4, i % 4].imshow(img, cmap='gray')
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
    plt.show()

def predict(images, path_to_model):
    model = tf.keras.models.load_model(path_to_model)
    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    
    ds_config = utils.load_dataset_config(config.data_path)
    max_len = ds_config['max_word_length']
    image_size = ds_config['image_size']
    
    dataset = OCRDataset(**ds_config)
    num_to_char = dataset.num_to_char

    for i, image in enumerate(images):
        images[i] = dataset.add_padding_to_image(image, image_size)
    images = tf.stack([*images])
    
    preds = prediction_model.predict(images)
    pred_texts = decode_batch_predictions(preds, max_len, num_to_char)
    return images, pred_texts

def inference(path_to_images, path_to_model=config.path_to_model, viz=True):
    name_images = os.listdir(path_to_images)    
    paths = [os.path.join(path_to_images, name) for name in name_images]
    np.random.shuffle(paths)
    # paths = paths[:16]
    images = []
    for path in paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, 1).numpy() / 255.
        images.append(image)

    images, predictions = predict(images, path_to_model)
    if viz: visualize_predictions(images, predictions)
    
    return images, predictions