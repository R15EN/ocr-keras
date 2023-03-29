import recognition.scripts.config as config
from recognition.scripts.utils import load_dataset_config, decode_predictions, visualize_predictions
from recognition.scripts.dataset import OCRDataGenerator

import os
import tensorflow as tf

def predict(images, path_to_model):
    model = tf.keras.models.load_model(path_to_model)
    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    
    ds_config = load_dataset_config(config.data_path)
    image_size = ds_config['image_size']
    max_len = ds_config['max_text_len']
    char_set = ds_config['char_set']

    for i, image in enumerate(images):
        images[i] = OCRDataGenerator.distortion_free_resize(image, image_size)
    images = tf.stack([*images])
    
    preds = prediction_model.predict(images)
    pred_texts = decode_predictions(preds, max_len, char_set)
    
    return images, pred_texts

def inference(path_to_images, path_to_model=config.path_to_model, viz=True):
    name_images = os.listdir(path_to_images)    
    paths = [os.path.join(path_to_images, name) for name in name_images]
    
    paths = paths[:16]
    images = []
    for path in paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, 3).numpy() / 255.
        images.append(image)

    images, predictions = predict(images, path_to_model)
    
    if viz: visualize_predictions(images, predictions)
    
    return images, predictions