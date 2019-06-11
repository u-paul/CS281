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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from PIL import Image
import os
os.environ['KERAS_BACKEND']='theano'

# In[13]:

def binaryclassify():
  infected_train = os.listdir('../data/p_data/cell_images/infected/simple_train/') 
  uninfected_train = os.listdir('../data/p_data/cell_images/uninfected/simple_train')

  infected_test = os.listdir('../data/p_data/cell_images/infected/simple_test/') 
  uninfected_test = os.listdir('../data/p_data/cell_images/uninfected/simple_test/')

  # In[14]:


  data_train = []
  labels_train = []

  for i in infected_train:
      try:
          image = cv2.imread("../data/p_data/cell_images/infected/simple_train/"+i)
          image_array = Image.fromarray(image , 'RGB')
          resize_img = image_array.resize((50 , 50))
          data_train.append(np.array(resize_img))
          labels_train.append(1)
      except AttributeError:
          continue

  for u in uninfected_train:
      try:
          image = cv2.imread("../data/p_data/cell_images/uninfected/simple_train/"+u)
          image_array = Image.fromarray(image , 'RGB')
          resize_img = image_array.resize((50 , 50))
          data_train.append(np.array(resize_img))
          labels_train.append(0)
      except AttributeError:
          continue

  cells_train = np.array(data_train)
  labels_train = np.array(labels_train)

  np.save('Cells_train' , cells_train)
  np.save('Labels_train' , labels_train)




  data_test = []
  labels_test = []

  for i in infected_test:
      try:
          image = cv2.imread("../data/p_data/cell_images/infected/simple_test/"+i)
          image_array = Image.fromarray(image , 'RGB')
          resize_img = image_array.resize((50 , 50))
          data_test.append(np.array(resize_img))
          labels_test.append(1)
      except AttributeError:
          continue

  for u in uninfected_test:
      try:
          image = cv2.imread("../data/p_data/cell_images/uninfected/simple_test/"+u)
          image_array = Image.fromarray(image , 'RGB')
          resize_img = image_array.resize((50 , 50))
          data_test.append(np.array(resize_img))
          labels_test.append(0)
      except AttributeError:
          continue

  cells_test = np.array(data_test)
  labels_test = np.array(labels_test)

  np.save('Cells_test' , cells_test)
  np.save('Labels_test' , labels_test)


  n_train = np.arange(cells_train.shape[0])
  np.random.shuffle(n_train)
  cells_train = cells_train[n_train]
  labels_train = labels_train[n_train]


  n_test = np.arange(cells_test.shape[0])
  np.random.shuffle(n_test)
  cells_test = cells_test[n_test]
  labels_test = labels_test[n_test]





  # In[18]:


  cells_train = cells_train.astype(np.float32)
  labels_train = labels_train.astype(np.int32)
  cells_train = cells_train/255


  cells_test = cells_test.astype(np.float32)
  labels_test = labels_test.astype(np.int32)
  cells_test = cells_test/255




  # In[19]:


  from sklearn.model_selection import train_test_split

  train_x, train_y = cells_train, labels_train

  eval_x , test_x , eval_y , test_y = train_test_split(cells_test , labels_test , 
                                                      test_size = 0.6 , 
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




  save_dir = os.path.join(os.getcwd(), 'saved_models')
  model_name = 'keras_mornot_trained_model.h5'


  # In[42]:


  model = Sequential()
  model.add(Conv2D(256, (7, 7), padding='same',
                   input_shape= train_x.shape[1:]))
  model.add(Activation('relu'))

  model.add(Conv2D(128, (3, 3)))
  model.add(Activation('relu'))


  model.add(Conv2D(64, (5, 5)))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))


  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

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
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
                                                                                                                                                      




  model.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=10,
                validation_data=(eval_x, eval_y),
                shuffle=True)



  y_pred = model.predict(test_x)
  y_pred = np.argmax(y_pred, axis = 1)


  test_y = np.argmax(test_y, axis = 1)
  print(accuracy_score(y_pred, test_y))
  print(classification_report(test_y, y_pred))




def cellclass():
  g_train = os.listdir('../data/p_data/types/gametocyte/train/') 
  r_train = os.listdir('../data/p_data/types/ring/train/') 
  t_train = os.listdir('../data/p_data/types/trophozoite/train/') 
  s_train = os.listdir('../data/p_data/types/schizont/train/') 
  d_train = os.listdir('../data/p_data/types/difficult/train/') 

  train_dict = {'gametocyte' : g_train, 'ring': r_train, 'trophozoite': t_train,'schizont' :s_train, 'difficult': d_train}


  g_test = os.listdir('../data/p_data/types/gametocyte/test/') 
  r_test = os.listdir('../data/p_data/types/ring/test/') 
  t_test = os.listdir('../data/p_data/types/trophozoite/test/') 
  s_test = os.listdir('../data/p_data/types/schizont/test/') 
  d_test = os.listdir('../data/p_data/types/difficult/test/') 

  test_dict = {'gametocyte' : g_test, 'ring' : r_test, 'trophozoite': t_test, 'schizont': s_test, 'difficult':d_test}

  # In[14]:


  label_dict = {'gametocyte': 1, 'ring':2, 'trophozoite':3, 'schizont':4, 'difficult':0}


  data_train = []
  labels_train = []


  data_test = []
  labels_test = []


  for key, val in train_dict.items():
    for i in val:
      try:
          # print(key, i)
          
          image = cv2.imread("../data/p_data/types/"+key+"/train/"+i)
          image_array = Image.fromarray(image , 'RGB')
          resize_img = image_array.resize((50 , 50))
          data_train.append(np.array(resize_img))
          labels_train.append(label_dict[key])
      except AttributeError:
          continue

  print(len(data_train))
  # exit()
  for key, val in test_dict.items():
    for i in val:
      try:
          image = cv2.imread("../data/p_data/types/"+key+"/test/"+i)
          image_array = Image.fromarray(image , 'RGB')
          resize_img = image_array.resize((50 , 50))
          data_test.append(np.array(resize_img))
          labels_test.append(label_dict[key])
      except AttributeError:
          continue

  cells_train = np.array(data_train)
  labels_train = np.array(labels_train)

  np.save('Cells_train' , cells_train)
  np.save('Labels_train' , labels_train)


  n_train = np.arange(cells_train.shape[0])
  np.random.shuffle(n_train)
  cells_train = cells_train[n_train]
  labels_train = labels_train[n_train]




  cells_test = np.array(data_test)
  labels_test = np.array(labels_test)

  np.save('Cells_test' , cells_test)
  np.save('Labels_test' , labels_test)
  n_test = np.arange(cells_test.shape[0])
  np.random.shuffle(n_test)
  cells_test = cells_test[n_test]
  labels_test = labels_test[n_test]





  # In[18]:


  cells_train = cells_train.astype(np.float32)
  labels_train = labels_train.astype(np.int32)
  cells_train = cells_train/255


  cells_test = cells_test.astype(np.float32)
  labels_test = labels_test.astype(np.int32)
  cells_test = cells_test/255




  # In[19]:


  from sklearn.model_selection import train_test_split

  train_x, train_y = cells_train, labels_train

  eval_x , test_x , eval_y , test_y = train_test_split(cells_test , labels_test , 
                                                      test_size = 0.6 , 
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
  num_classes = 5
  epochs = 100
  data_augmentation = False
  num_predictions = 20


  # In[33]:


  train_y = keras.utils.to_categorical(train_y, num_classes)
  eval_y= keras.utils.to_categorical(eval_y,num_classes)
  test_y = keras.utils.to_categorical(test_y,num_classes)




  save_dir = os.path.join(os.getcwd(), 'saved_models')
  model_name = 'keras_mornot_trained_model.h5'


  # In[42]:


  model = Sequential()
  model.add(Conv2D(128, (7, 7), padding='same',
                   input_shape= train_x.shape[1:]))
  model.add(Activation('relu'))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(32, (5, 5)))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))





  model.add(Flatten())
            
  model.add(Dense(2000))
  model.add(Activation('relu'))
     
  # model.add(Dense(1000))
  # model.add(Activation('relu'))          
      
  # model.add(Dense(500))
  # model.add(Activation('relu'))           

  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

  model.compile(loss='categorical_crossentropy',
                optimizer=opt,   
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
                                                                                                                                                      




  model.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=10,
                validation_data=(eval_x, eval_y),
                shuffle=True)



  y_pred = model.predict(test_x)
  y_pred = np.argmax(y_pred, axis = 1)


  test_y = np.argmax(test_y, axis = 1)
  print(accuracy_score(y_pred, test_y))
  print(classification_report(test_y, y_pred))

cellclass()