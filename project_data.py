import numpy as np
import time
import sys
import os
import random
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt
from shutil import copyfile

import cv2
import tensorflow as tf



# df = pd.read_csv('../data/p_data/train.csv', skiprows= 1)

# df_train = df.iloc[:2154]
# print(df_train.shape)
# df_test = df.iloc[2154:]
# print(df_test.shape)

# df_train.to_csv('../data/p_data/train1.csv', index = False)
# df_test.to_csv('../data/p_data/test.csv', index = False)
base_path= '../data/p_data'
train_df = pd.read_csv('../data/p_data/simple_train.csv')
f = open(base_path + "/annotation.txt","w+")
for idx, row in train_df.iterrows():
#     sys.stdout.write(str(idx) + '\r')
#     sys.stdout.flush()
	# img = cv2.imread((row['FileName']))
	# height, width = img.shape[:2]

	# print(height, width)
	# print()
	x1 = int(row['CMin'] * 1)
	x2 = int(row['CMax'] * 1)
	y1 = int(row['RMin'] * 1)
	y2 = int(row['RMax'] * 1)

	# data_path = '../data/p_data/train'
	fileName = (row['FileName'])
	className = row['ClassName']
	f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
# f.close()
test_df = pd.read_csv('../data/p_data/simple_test.csv')

f.close()
# For test
f= open(base_path + "/test_annotation.txt","w+")
for idx, row in test_df.iterrows():
	sys.stdout.write(str(idx) + '\r')
	sys.stdout.flush()
	try:
		# img = cv2.imread((row['FileName']))
		# height, width = img.shape[:2]
		x1 = int(row['CMin'] * 1)
		x2 = int(row['CMax'] * 1)
		y1 = int(row['RMin'] * 1)
		y2 = int(row['RMax'] * 1)
	    
		# data_path = '../data/p_data/test'
		fileName = (row['FileName'])
		className = row['ClassName']
		f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
	except Exception as e:
		print(e)


f.close()