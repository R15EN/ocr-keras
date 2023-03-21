from tensorflow.keras.utils import Sequence
import numpy as np
import cv2

class OCRDataGenerator(Sequence):
    def __init__(self, img_paths, labels, batch_size, image_size=(256, 64), max_text_len=30, char_set=None, pad_token=None):
        self.img_paths = img_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = image_size
        self.max_text_len = max_text_len
        self.char_set = char_set or self._get_char_set()
        self.pad_token = pad_token if pad_token else len(self.char_set) + 5 
        self.indexes = np.arange(len(self.img_paths))

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_img_paths = [self.img_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        batch_images = np.zeros((len(batch_img_paths), *self.img_size, 3), dtype=np.float32)
        batch_texts = []
        for i, path in enumerate(batch_img_paths):
            img = cv2.imread(path)[...,::-1]
            img = self.distortion_free_resize(img, self.img_size)
            batch_images[i] = img.astype(np.float32) / 255.0
            batch_texts.append(batch_labels[i]) 
        batch_texts = self._encode_text(batch_texts)
        batch_texts = np.stack(batch_texts)
        inputs = {"image": batch_images, "label": batch_texts}

        return inputs

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def _get_char_set(self):
        char_set = set()
        for label in self.labels:
            char_set.update(set(label))
        return sorted(list(char_set) + [''])

    def _encode_text(self, texts):
        char_map = {c: i for i, c in enumerate(self.char_set)}
        encoded_texts = np.full((len(texts), self.max_text_len), self.pad_token)
        for i, text in enumerate(texts):
            for j, c in enumerate(text):
                if j == self.max_text_len:
                    break
                encoded_texts[i, j] = char_map[c]
        return encoded_texts
        
    def distortion_free_resize(self, image, img_size):
        width, height = img_size
        base_width, base_height = image.shape[1], image.shape[0]
        top_pad, bot_pad = 0, 0
        left_pad, right_pad = 0, 0
        if (width / base_width) < (height / base_height):
            ratio = width / base_width
            image = cv2.resize(image, (width, int(base_height*ratio)))
            top_pad = (height - image.shape[0]) // 2
            bot_pad = height - image.shape[0] - top_pad
        else:
            ratio = height / base_height
            image = cv2.resize(image, (int(base_width*ratio), height))
            left_pad = (width - image.shape[1]) // 2
            right_pad = width - image.shape[1] - left_pad
            
        image = cv2.copyMakeBorder(
            image, 
            top_pad, bot_pad, 
            left_pad, right_pad, 
            cv2.BORDER_CONSTANT
        )
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  
         
        return image