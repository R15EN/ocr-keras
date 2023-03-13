import os
import numpy as np
import tensorflow as tf

import recognition.scripts.config as config
import recognition.scripts.utils as utils
import recognition.scripts.metrics as metrics
from recognition.scripts.model import ocr_model
from recognition.scripts.dataset import OCRDataset

class Callback(tf.keras.callbacks.Callback):
    def __init__(self, pred_model, validation_images, validation_labels, max_len, num_to_char):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.max_len = max_len
        self.num_to_char = num_to_char
    
    def on_epoch_end(self, epoch, logs=None):
        pred_texts = []
        true_texts = []
        
        labels = self.num_to_char(self.validation_labels)
        predictions = self.prediction_model.predict(self.validation_images)
        predictions = self.decode_predictions(predictions)
        
        for i in range(len(labels)):
            label = tf.strings.reduce_join(labels[i]) \
                .numpy() \
                .decode('utf-8') \
                .replace('[UNK]', '')
            pred_texts.append(predictions[i])
            true_texts.append(label)
            
        print(
            f"CER for epoch {epoch + 1}: {metrics.cer(true_texts, pred_texts):.4f}", 
            f"WER for epoch {epoch + 1}: {metrics.wer(true_texts, pred_texts):.4f}",
            sep='\n'
        )  

    def decode_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]

        results = tf.keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True)[0][0][:, :self.max_len]

        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

def train_model(save_path=config.path_to_model, 
                path_to_images=config.path_to_images, 
                path_to_labels=config.path_to_labels):
    
    paths_and_labels = utils.load_data_from_file(path_to_labels, shuffle=True)
    paths_and_labels = [
        tuple([os.path.join(path_to_images, path.split('/', 1)[1]), label]) 
        for path, label in paths_and_labels
    ] 
    # paths_and_labels = paths_and_labels[:80000]
    
    train_samples, validation_samples = \
        utils.split_data(paths_and_labels, test_size=0.1)
    validation_samples, test_samples = \
        utils.split_data(validation_samples, test_size=0.5)

    train_image_paths, train_labels = \
        utils.get_paths_and_labels(train_samples)
    validation_image_paths, validation_labels = \
        utils.get_paths_and_labels(validation_samples)
    test_image_paths, test_labels = \
        utils.get_paths_and_labels(test_samples)

    characters = sorted(set(''.join(train_labels)))
    max_len = len(max(train_labels, key=len))
    
    ds_params = {'vocabulary': characters, 
                 'max_word_length': max_len,
                 'image_size': config.image_size,
                 'batch_size': config.batch_size} 
    dataset = OCRDataset(**ds_params)

    train_ds = dataset.prepare_dataset(train_image_paths, train_labels)
    validation_ds = dataset.prepare_dataset(validation_image_paths, validation_labels)
    test_ds = dataset.prepare_dataset(test_image_paths, test_labels)
    
    char_to_num = dataset.char_to_num
    num_to_char = dataset.num_to_char

    validation_images = []
    validation_labels = []
    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])    
    validation_images = tf.concat([*validation_images], axis=0)
    validation_labels = tf.concat([*validation_labels], axis=0)
    
    model = ocr_model(*config.image_size, char_to_num)
    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    
    callback = Callback(
        prediction_model, 
        validation_images,
        validation_labels,
        max_len,
        num_to_char
    )
    
    # Train the model.
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=config.epochs,
        callbacks=callback,
    )
    
    print(model.evaluate(test_ds))
    
    model.save(save_path)
    utils.save_dataset_config(config.data_path, ds_params)