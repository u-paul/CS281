#!/usr/bin/env python
# coding: utf-8

# In[41]:


from __future__ import absolute_import, division, print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
import seaborn as sns
import keras_metrics
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image
import os


# In[13]:


infected = os.listdir('../data/p_data/cell_images/Parasitized/') 
uninfected = os.listdir('../data/p_data/cell_images/Uninfected/')


# In[14]:


data = []
labels = []

for i in infected:
    try:
        image = cv2.imread("../data/p_data/cell_images/Parasitized/"+i)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((50 , 50))
        data.append(np.array(resize_img))
        labels.append(1)
    except AttributeError:
        continue

for u in uninfected:
    try:
        image = cv2.imread("../data/p_data/cell_images/Uninfected/"+u)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((50 , 50))
        data.append(np.array(resize_img))
        labels.append(0)
    except AttributeError:
        continue
cells = np.array(data)
labels = np.array(labels)

np.save('Cells' , cells)
np.save('Labels' , labels)

# print('Cells : {} | labels : {}'.format(cells.shape , labels.shape))


# In[15]:


plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(49):
    n += 1 
    r = np.random.randint(0 , cells.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(cells[r[0]])
    plt.title('{} : {}'.format('Infected' if labels[r[0]] == 1 else 'Unifected' ,
                               labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
# plt.show()


# In[16]:


plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(cells[0])
plt.title('Infected Cell')
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(cells[10])
plt.title('Uninfected Cell')
plt.xticks([]) , plt.yticks([])

# plt.show()


# In[17]:


n = np.arange(cells.shape[0])
np.random.shuffle(n)
cells = cells[n]
labels = labels[n]


# In[18]:


cells = cells.astype(np.float32)
labels = labels.astype(np.int32)
cells = cells/255


# In[19]:


from sklearn.model_selection import train_test_split

train_x , x , train_y , y = train_test_split(cells , labels , 
                                            test_size = 0.2 ,
                                            random_state = 111)

eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                    test_size = 0.5 , 
                                                    random_state = 111)


# In[22]:


plt.figure(1 , figsize = (15 ,5))
n = 0 
for z , j in zip([train_y , eval_y , test_y] , ['train labels','eval labels','test labels']):
    n += 1
    plt.subplot(1 , 3  , n)
    sns.countplot(x = z )
    plt.title(j)
# plt.show()


# In[30]:


print('train data shape {} ,eval data shape {} , test data shape {}'.format(train_x.shape,
                                                                           eval_x.shape ,
                                                                           test_x.shape))


# In[31]:


batch_size = 32
num_classes = 2
epochs = 100
data_augmentation = False
num_predictions = 20


# In[33]:


train_y = keras.utils.to_categorical(train_y, num_classes)
eval_y= keras.utils.to_categorical(eval_y,num_classes)
test_y = keras.utils.to_categorical(test_y,num_classes)

# In[ ]:





# In[ ]:





# In[24]:





# In[34]:


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mornot_trained_model.h5'


# In[42]:


model = Sequential()
model.add(Conv2D(32, (7, 7), padding='same',
                 input_shape= train_x.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))


model.add(Conv2D(10, (5, 5)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))


model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
          
model.add(Dense(2000))
model.add(Activation('relu'))
   
model.add(Dense(1000))
model.add(Activation('relu'))          
    
model.add(Dense(500))
model.add(Activation('relu'))           

model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,   
                metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])
                                                                                                                                                    


# In[ ]:


model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(eval_x, eval_y),
              shuffle=True)



scores = model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




