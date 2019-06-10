import pandas as pd
import numpy as np
import os, random

import cv2
from PIL import Image

import matplotlib.pyplot as plt
# %matplotlib inline

from skimage.io import imread, imshow
from skimage.transform import resize

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

random.seed()

csv_path        = '../data/p_data'

def parse_uninfected(images_csv='simple_test'):
	full_csv_path = os.path.join(csv_path, images_csv+'.csv')

	df = pd.read_csv(full_csv_path)
	df_uninfected = df.loc[df['ClassName'] == 'uninfected']
	print(df_uninfected.head())
	i = 1
	for idx, row in df_uninfected.iterrows():
		img = cv2.imread((row['FileName']))
		# height, width = 110,130
		x1 = int(row['CMin'] * 1)
		x2 = int(row['CMax'] * 1)
		y1 = int(row['RMin'] * 1)
		y2 = int(row['RMax'] * 1)
		cell = img[y1:y2, x1:x2]

		img_path = os.path.join(csv_path, 'uninfected', images_csv, str(i))
		cv2.imwrite(img_path+'.png', cell)

		i += 1
		# plt.imshow(cell)
		# plt.show()
		if (i % 100 == 0):
			print('{:<10} {}'.format('Finished', str(i)))
		if (i == 13000):
			return


def parse_infected(images_csv='simple_test'):
	full_csv_path = os.path.join(csv_path, images_csv+'.csv')

	df = pd.read_csv(full_csv_path)
	df_infected = df.loc[df['ClassName'] == 'infected']
	print(df_infected.head())
	i = 1
	for idx, row in df_infected.iterrows():
		img = cv2.imread((row['FileName']))
		# height, width = 110,130
		x1 = int(row['CMin'] * 1)
		x2 = int(row['CMax'] * 1)
		y1 = int(row['RMin'] * 1)
		y2 = int(row['RMax'] * 1)
		cell = img[y1:y2, x1:x2]

		r90, r180, bl = augment_image(cell)
		mv_cell      = scale_and_move(img, x1, x2, y1, y2)

		img_path = os.path.join(csv_path, 'infected', images_csv, str(i))
		cv2.imwrite(img_path+'.png', cell)
		cv2.imwrite(img_path+'_90.png', r90)
		cv2.imwrite(img_path+'_1800.png', r180)
		cv2.imwrite(img_path+'_bl.png', bl)
		cv2.imwrite(img_path+'_mv.png', mv_cell)

		i += 1
		# plt.imshow(cell)
		# plt.show()
		if (i % 100 == 0):
			print('{:<10} {}'.format('Finished', str(i)))


def augment_image(img_array):
	img = Image.fromarray(img_array, 'RGB')

	rotated90  = img.rotate(90)
	rotated180 = img.rotate(180)
	blur       = cv2.blur(np.array(img) ,(10,10))

	return (np.array(rotated90), np.array(rotated180), np.array(blur))


def scale_and_move(img, x1, x2, y1, y2):
	try:
		scale = random.randint(40,70)
		move  = random.randint(-40, 40)
		
		new_y1 = y1-scale+move
		if new_y1 < 0: new_y1 = 0
		new_y2 = y2+scale+move
		if new_y2 < 0: new_y2 = 0
		new_x1 = x1-scale+move
		if new_x1 < 0: new_x1 = 0
		new_x2 = x2+scale+move
		if new_x2 < 0: new_x2 = 0

		mv_cell = img[y1-scale+move:y2+scale+move, x1-scale+move:x2+scale+move]
	except Exception as e:
		print("Exception: {}".format(e))
		exit()
	return mv_cell


def main():
	print("Splitting Infected:")
	parse_infected('simple_train')
	print("\n\nSplitting Uninfected:")
	parse_uninfected('simple_train')


if __name__ == "__main__":
	main()