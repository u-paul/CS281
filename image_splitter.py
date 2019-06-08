import pandas as pd
import numpy as np
import os

import cv2

import matplotlib.pyplot as plt
# %matplotlib inline

from skimage.io import imread, imshow
from skimage.transform import resize

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/p_data/simple_test.csv')


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
	path = ('../data/p_data/infected')
	cv2.imwrite(os.path.join(path , str(i)+'.png'), cell)
	i += 1
	# plt.imshow(cell)
	# plt.show()
	print('Done '+ str(i))