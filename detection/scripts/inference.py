from detection.scripts.model import pixel_link_model
from detection.scripts.utils import mask_to_bboxes, resize_image
from detection.scripts.decode import decode_batch
from detection.scripts.weights import set_weights_and_biases
import detection.scripts.config as config

import cv2
import numpy as np
from tensorflow.nn import softmax

def create_model(image_shape, h5=True):
    model = pixel_link_model(image_shape)
    if h5: 
        model.load_weights(config.weight_path)
    else: 
        _ = model(np.random.normal(size=(1, *image_shape)))
        set_weights_and_biases(model, config.ckpt_path)
        
    return model

def predict(image):
    image = resize_image(image)
    model = create_model(image.shape)
    
    img_col_corr = image - config.rgb_mean
    cls_scores, link_scores = model.predict(img_col_corr[None, ...])

    cls_scores = softmax(cls_scores).numpy()
    link_scores = softmax(link_scores.reshape((*link_scores.shape[:-1], 8, 2))).numpy()

    masks = decode_batch(cls_scores, link_scores, 
                         pixel_conf_threshold=0.6, 
                         link_conf_threshold=0.9)
    
    bboxes = mask_to_bboxes(masks[0], image.shape)
    if len(bboxes):
        bboxes = np.reshape(bboxes, (len(bboxes), -1, 2))

    return image, bboxes

def inference(path_to_image, viz=True):    
    image = cv2.imread(path_to_image)[...,::-1]

    image, bboxes = predict(image)
    
    for bbox in bboxes:
        cv2.drawContours(image, [bbox], 0, (0, 0, 255), 2)
        
    if viz:
        cv2.imshow('image', image[...,::-1])
        cv2.waitKey(0)

    return image

    


    

    
