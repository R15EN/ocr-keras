import os
import gc
import tensorflow as tf

import recognition.scripts.config as config
import recognition.scripts.utils as utils
import recognition.scripts.metrics as metrics
from recognition.scripts.model import ocr_model
from recognition.scripts.dataset import OCRDataGenerator

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, pred_model, validation_ds, char_set, max_len, pad_token):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_ds = validation_ds
        self.char_set = char_set
        self.max_len = max_len
        self.pad = pad_token
        self.pred_texts = []
        self.gt_texts = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.pred_texts = []
        self.gt_texts = []
        
        for batch in self.validation_ds:
            predictions = self.prediction_model.predict(batch['image'], verbose=0)
            predictions = utils.decode_predictions(predictions, self.max_len, self.char_set)
            
            for i, label in enumerate(batch['label']):
                label = tf.gather(label, tf.where(tf.math.not_equal(label, self.pad)))
                label = tf.reshape(label, [-1])
                label = ''.join([self.char_set[int(j)] for j in label])
                self.pred_texts.append(predictions[i])
                self.gt_texts.append(label)
                
        cer_score = metrics.cer(self.gt_texts, self.pred_texts)
        wer_score = metrics.wer(self.gt_texts, self.pred_texts)

        print(
            f"CER for epoch {epoch + 1}: {cer_score:.4f}",
            f"WER for epoch {epoch + 1}: {wer_score:.4f}",
            sep='\n'
        )
        
        gc.collect()
        tf.keras.backend.clear_session()

def train_model(save_path=config.path_to_model, 
                path_to_images=config.path_to_images, 
                path_to_labels=config.path_to_labels):
    
    paths_and_labels = utils.load_data_from_file(path_to_labels, shuffle=True)
    paths_and_labels = [
        tuple([os.path.join(path_to_images, path.split('/', 1)[1]), label]) 
        for path, label in paths_and_labels
    ] 
    
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

    char_set = [''] + sorted(set(''.join(train_labels)))
    max_len = len(max(train_labels, key=len))
    pad_token = len(char_set) + 5
    
    ds_params = {
        'batch_size': config.batch_size,
        'image_size': config.image_size,
        'max_text_len': max_len,
        'char_set': char_set,
        'pad_token': pad_token
    } 

    train_ds = OCRDataGenerator(train_image_paths, train_labels, **ds_params)
    validation_ds = OCRDataGenerator(validation_image_paths, validation_labels, **ds_params)
    test_ds = OCRDataGenerator(test_image_paths, test_labels, **ds_params)
    
    model = ocr_model(*config.image_size, char_set=char_set)
    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    
    metrics_callback = MetricsCallback(
        prediction_model, 
        validation_ds,
        char_set,
        max_len,
        pad_token
    )
    
    # Train the model.
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=config.epochs,
        callbacks=metrics_callback,
    )
    
    print('Test data evaluate:', model.evaluate(test_ds))
    
    model.save(save_path)
    utils.save_dataset_config(config.data_path, ds_params)