import tensorflow as tf
from tensorflow.keras.layers import StringLookup

class OCRDataset():
    def __init__(self, vocabulary, max_word_length, image_size, batch_size):
        self.padding_token = len(vocabulary) + 5
        self.max_word_length = max_word_length
        self.image_size = image_size
        self.batch_size = batch_size
        
        self._char_to_num = StringLookup(vocabulary=list(vocabulary), mask_token=None)
        self._num_to_char = StringLookup(
            vocabulary=self._char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
    
    def add_padding_to_image(self, image, image_size):
        w, h = image_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        h_pad = h - tf.shape(image)[0]
        w_pad = w - tf.shape(image)[1]
        
        top_pad = h_pad // 2
        bot_pad = h_pad - top_pad
        left_pad = w_pad // 2
        right_pad = w_pad - left_pad
        paddings = [[top_pad, bot_pad],
                    [left_pad, right_pad],
                    [0, 0]]
        
        image = tf.pad(image, paddings=paddings)
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image

    def preprocess_image(self, image_path):    
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 3)
        image = self.add_padding_to_image(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.
        return image

    def vectorize_label(self, label):
        label = self._char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_word_length - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=self.padding_token )
        return label

    def process_images_labels(self, image_path, label):
        image = self.preprocess_image(image_path)
        label = self.vectorize_label(label)
        return {"image": image, "label": label}

    def prepare_dataset(self, image_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)) \
            .map(self.process_images_labels, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(self.batch_size) \
            .cache() \
            .prefetch(tf.data.AUTOTUNE)
        return dataset

    @property
    def char_to_num(self):
        return self._char_to_num
    
    @property
    def num_to_char(self):
        return self._num_to_char