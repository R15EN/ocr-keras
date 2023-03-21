from detection.scripts.inference import predict as det_predict, \
                                        inference as det_inference
from recognition.scripts.inference import predict as ocr_predict,\
                                          inference as ocr_inference
import detection.scripts.config as det_config
import recognition.scripts.config as ocr_config

from recognition.scripts.train import train_model

import argparse
import numpy as np
import cv2

def coordinate_transform(coords):
    boxes = []
    for c in coords:
        x_min, x_max = min(c[:, 0]), max(c[:, 0])
        y_min, y_max = min(c[:, 1]), max(c[:, 1])
        boxes.append([x_min, y_min, x_max-x_min, y_max-y_min])
    boxes = np.asarray(boxes)
    return boxes

def sort_boxes(boxes):
    max_height = np.mean(boxes[::, 3])
    by_y = sorted(boxes, key=lambda x: x[1])

    line_y = by_y[0][1] 
    line = 1
    by_line = []
    
    for x, y, w, h in by_y:
        if y > line_y + max_height:
            line_y = y
            line += 1
        by_line.append((line, x, y, w, h))
        
    boxes_sorted = sorted(by_line)
    return boxes_sorted

def prediction(path_to_image, path_to_model):
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, bboxes = det_predict(image)

    boxes = coordinate_transform(bboxes)
    boxes = sort_boxes(boxes)
    
    crops = []
    lines = []
    for i, box in enumerate(boxes):
        line, x, y, w, h = box
        crop = image[y: y+h, x: x+w, :] / 255.
        lines.append(line)
        crops.append(crop) 
        
    _, predictions = ocr_predict(crops, path_to_model)
    
    for i in range(1, len(lines)):
        if lines[i] != lines[i-1]:
            predictions[i] = '\n' + predictions[i] 
    
    predictions = ' '.join(predictions)
    return predictions

def main(args):
    if args.predict and args.image:
        result = prediction(args.image, args.model)
        print(result)
    elif args.detect and args.image:
        det_inference(args.image)
    elif args.ocr:
        ocr_inference(args.path_to_images, args.model)
    elif args.ocr_train:
        train_model(args.model, args.path_to_images, args.path_to_labels)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p', 
        '--predict',
        action='store_true',
        help='Search and text recognition'    
    )

    parser.add_argument(
        '-d', 
        '--detect',
        action='store_true',
        help='Detection model inference'
    )
    
    parser.add_argument(
        '-i',
        '--image',
        type=str,
        help='Path to image for text detection and recognition (or only detection)'
    )
    
    parser.add_argument(
        '-o', 
        '--ocr', 
        action='store_true',
        help='OCR model inference'
    )
    
    parser.add_argument(
        '-t', 
        '--ocr_train', 
        action='store_true',
        help='OCR model training'
    )
    
    parser.add_argument(
        '-pi',
        '--path_to_images', 
        type=str, 
        default=ocr_config.path_to_images,
        help='Path to images for OCR model inference or model training. if you want to do inference, you need to specify the path to a directory containing only images'
    )
    
    parser.add_argument(
        '-pl',
        '--path_to_labels', 
        type=str, 
        default=ocr_config.path_to_labels,
        help='Path to labels for OCR model training'
    )
    
    parser.add_argument(
        '-m',
        '--model', 
        type=str, 
        default=ocr_config.path_to_model,
        help='The save path for the trained model, or the path if you want to load the model'
    )
    
    args = parser.parse_args()
    main(args)
 