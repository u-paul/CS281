import numpy as np
import json, os
import time, sys, random
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt
from shutil import copyfile

import cv2
import tensorflow as tf

images_path         = 'malaria/images/'
test_json_path      = 'malaria/test.json'
training_json_path  = 'malaria/training.json'

'''
1328 Images

JSON Attributes:
data[n] gives you the nth image
    data[n] has two keys: 'image' and 'objects'

data[n]['image'] has three keys
    ['checksum']
    ['pathname'] == returns path of image
        path is '/image/xxxxx.png'
    ['shape'] has three keys
        'r', 'c', 'channels'
        
data[n]['objects'] has two keys
    ['bounding_box'] has two keys
        ['minimum']
            'r', 'c'
        ['maximum']
            'r', 'c'
    ['category']
'''

def load_json(path):
    with open(path, 'r') as jf:
        data = json.load(jf)

    return data


def plot_bounding_box(data, img_index):
    img_path = data[img_index]['image']['pathname']
    img_path = 'malaria' + img_path
    img = io.imread(img_path)

    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    plt.title('Original Image')
    plt.imshow(img)

    img_bbox = img.copy()
    objects = data[img_index]['objects']
    for object in objects:
        label_name = object['category']
        if (label_name == 'red blood cell'):
            # Don't box healthy cells
            continue

        rmin = object['bounding_box']['minimum']['r']
        rmax = object['bounding_box']['maximum']['r']
        cmin = object['bounding_box']['minimum']['c']
        cmax = object['bounding_box']['maximum']['c']

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(img_bbox, (cmin,rmin), (cmax,rmax), (255,0,0), 4)
        cv2.putText(img_bbox, label_name, (cmin,rmin-10), font, 1, (255,0,0), 4,)
    
    plt.subplot(1,2,2)
    plt.title('Image with Bounding Box')
    plt.imshow(img_bbox)
    plt.show()

def main():
    training = load_json(training_json_path)

    plot_bounding_box(training, 900)

    

if __name__ ==  "__main__":
    main()