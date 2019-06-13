import numpy as np
import json, os
import time, sys, random
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
from shutil import copyfile

import cv2
import tensorflow as tf

images_path         = 'malaria/images/'
test_json_path      = 'malaria/test.json'
training_json_path  = 'malaria/training.json'

# Simple jsons are just infected vs uninfected
#   Removed categories
simple_train_path   = 'malaria/training_simple.json'
simple_test_path    = 'malaria/test_simple.json'

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


# Creates train and test csv files
#   Also creates annotation file (txt version)
def clean_data(simple=False):
    if simple:
        training = load_json(simple_train_path)
        testing  = load_json(simple_test_path)

        label_names = ['red blood cell', 'infected']
    else:
        training = load_json(training_json_path)
        testing  = load_json(test_json_path)

    train_df = pd.DataFrame(columns=['FileName', 'CMin', 'CMax', 'RMin', 'RMax', 'ClassName'])
    test_df  = pd.DataFrame(columns=['FileName', 'CMin', 'CMax', 'RMin', 'RMax', 'ClassName'])

    # Find boxes in each image and put them in a dataframe
    for item in training:
        for object in item['objects']:
            train_df = train_df.append({'FileName': 'malaria' + item['image']['pathname'],
                                        'CMin': object['bounding_box']['minimum']['c'],
                                        'CMax': object['bounding_box']['maximum']['c'],
                                        'RMin': object['bounding_box']['minimum']['r'],
                                        'RMax': object['bounding_box']['maximum']['r'],
                                        'ClassName': object['category']},
                                        ignore_index=True)

    for item in testing:
        for object in item['objects']:
            test_df = test_df.append({'FileName': 'malaria' + item['image']['pathname'],
                                        'CMin': object['bounding_box']['minimum']['c'],
                                        'CMax': object['bounding_box']['maximum']['c'],
                                        'RMin': object['bounding_box']['minimum']['r'],
                                        'RMax': object['bounding_box']['maximum']['r'],
                                        'ClassName': object['category']},
                                        ignore_index=True)

    if simple:
        train_csv = 'simple_train.csv'
        test_csv  = 'simple_test.csv'
    else:
        train_csv = 'train.csv'
        test_csv  = 'test.csv'

    train_df.to_csv(train_csv)
    test_df.to_csv(test_csv)

    # Write train.csv to annotation.txt
    train_df  = pd.read_csv(train_csv)

    # For training
    if simple:
        train_annotation = 'simple_annotation.txt'
    else:
        train_annotation = 'annotation.txt'
    with open(train_annotation, 'w+') as f:
        for idx, row in train_df.iterrows():
            img = cv2.imread(row['FileName'])
            c1 = int(row['CMin'])
            c2 = int(row['CMax'])
            r1 = int(row['RMin'])
            r2 = int(row['RMax'])

            fileName  = row['FileName']
            className = row['ClassName']
            f.write('{},{},{},{},{},{}\n'.format(fileName, c1, r1, c2, r2, className))

    # For testing
    if simple:
        test_annotation = 'simple_test_annotation.txt'
    else:
        test_annotation = 'test_annotation.txt'
    with open(test_annotation, 'w+') as f:
        for idx, row in test_df.iterrows():
            img = cv2.imread(row['FileName'])
            c1 = int(row['CMin'])
            c2 = int(row['CMax'])
            r1 = int(row['RMin'])
            r2 = int(row['RMax'])

            fileName  = row['FileName']
            className = row['ClassName']
            f.write('{},{},{},{},{},{}\n'.format(fileName, c1, r1, c2, r2, className))


def main():
    training = load_json(training_json_path)

    #plot_bounding_box(training, 900)
    clean_data()



if __name__ ==  "__main__":
    main()