import os
import cv2 
import sys
import h5py
import pickle
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from datetime import datetime
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_crossentropy
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout,  Flatten
from keras.layers import BatchNormalization


import tensorboard
import tensorflow as ts
import tensorflow.keras.callbacks 

import matplotlib.cm as cm

file_name = 'SynthText.h5'
db = h5py.File(file_name, 'r')
im_names = list(db['data'].keys())

additional_files = 'train.h5'
addtional_db = h5py.File(additional_files, 'r')
add_im_names = list(addtional_db['data'].keys())

im_size = 28
data_train = []
x_train = []
y_train = []
x_test = []
y_test = []

val_file_name = 'SynthText_val.h5'
val_db = h5py.File(val_file_name, 'r')
val_names = list(val_db['data'].keys())
val_data = []
x_val = []
y_val = []

font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']


def checkingData():
	fig, ax = plt.subplots(6, 6, figsize = (6,6))
	temp = 0
	#Showing all the 12 images 
	for i in range(6):
		for j in  range(6):
			imgByIndex(temp,ax[i,j], True)
			temp += 1
	plt.show()	

def imgByIndex(index, ax, b):
	if(b == True):
		t = x_train[index]
	else:
		t = x_val[index]
	#Showing the iamges as gray scale
	ax.imshow(t, cmap = 'gray', interpolation='nearest')
	ax.set_xticks([])
	ax.set_yticks([])
	
def checkingValData():
	fig, ax = plt.subplots(6, 6, figsize = (6,6))
	temp = 0
	#Showing all the 12 images 
	for i in range(6):
		for j in  range(6):
			imgByIndex(temp,ax[i,j], False)
			temp += 1
	plt.show()	


'''
im_char - the char bounding box of the latter we working on
b_inx - the index in the char boiunding box
img - gray scale of the original image
'''
def latterCut(im_char, b_inx, img):
	bb = im_char[:,:,b_inx]
	postion1 = np.float32([bb.T[0], bb.T[1], bb.T[3], bb.T[2]])
	postion2 = np.float32([[0,0],[im_size,0],[0,im_size],[im_size,im_size]])
	
	M = cv2.getPerspectiveTransform(postion1, postion2)		
	resulte = cv2.warpPerspective(img, M , (im_size, im_size))
	return resulte	
	
'''
im_char - the char bounding box of the latter we working on
gray_img - gray scale of the original image
im_font - the fonts of the text in the image
b - True if the data is part of the training data false if part of val data
'''
def addingData(im_char, gray_img, im_font, b):
	nC = im_char.shape[-1]
	for b_inx in range(nC):
		latter = latterCut(im_char, b_inx, gray_img)
		try:
			re_latter = cv2.resize(latter, (im_size, im_size))
			if(b == True):
				data_train.append([re_latter, font_name.index(im_font[b_inx].decode('UTF-8'))])
			else:
				val_data.append([re_latter, font_name.index(im_font[b_inx].decode('UTF-8'))])
		except Exception as e:
			print(e)
			pass

'''
creating all the data from the iamge database 
'''	
def createTrainingData():
	for im_name in im_names:
		img = db['data'][im_name][:]
		im_font = db['data'][im_name].attrs['font']
		im_char = db['data'][im_name].attrs['charBB']
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		addingData(im_char, gray_img, im_font, True)
	for im_name in add_im_names:
		img = addtional_db['data'][im_name][:]
		im_font = addtional_db['data'][im_name].attrs['font']
		im_char = addtional_db['data'][im_name].attrs['charBB']
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		addingData(im_char, gray_img, im_font, True)
	#---------------------create the validation data-----------------
	for val_name in val_names:
		val_img = val_db['data'][val_name][:]
		val_font = val_db['data'][val_name].attrs['font']
		val_char = val_db['data'][val_name].attrs['charBB']
		gray_img = cv2.cvtColor(val_img, cv2.COLOR_BGR2GRAY)
		addingData(val_char, gray_img, val_font, False)

'''
splititng the data traing for x_train, y_train and x_test, y_test
when x is between 0 to 1 in precent we want to split the data train
'''				
def splitData(x):
	i = 0
	for features, label in data_train:
		if(i < int(len(data_train)*x)):
			x_train.append(features)
			y_train.append(label)
			i = i + 1
		else:
			x_test.append(features)
			y_test.append(label)
			i = i + 1
	for features, label in val_data:
		x_val.append(features)
		y_val.append(label)

'''
save all the data to pickle file after all the manupltion on the data
'''
def saveAndCloseData():
	pickle_out = open("x_train.pickle", "wb")
	pickle.dump(x_train, pickle_out)
	pickle_out.close()
	
	pickle_out = open("x_test.pickle", "wb")
	pickle.dump(x_test, pickle_out)
	pickle_out.close()
	
	pickle_out = open("y_train.pickle", "wb")
	pickle.dump(y_train, pickle_out)
	pickle_out.close()
	
	pickle_out = open("y_test.pickle", "wb")
	pickle.dump(y_test, pickle_out)
	pickle_out.close()

	pickle_out = open("x_val.pickle", "wb")
	pickle.dump(x_val, pickle_out)
	pickle_out.close()
	
	pickle_out = open("y_val.pickle", "wb")
	pickle.dump(y_val, pickle_out)
	pickle_out.close()
	
createTrainingData()

random.shuffle(data_train)
random.shuffle(val_data)

splitData(0.8)


x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], im_size, im_size, 1)
x_train = x_train.astype('float32')
x_train /= 255

x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], im_size, im_size, 1)
x_test = x_test.astype('float32')
x_test /= 255

x_val = np.asarray(x_val)
x_val = x_val.reshape(x_val.shape[0], im_size, im_size, 1)
x_val = x_val.astype('float32')
x_val /= 255

print('x_train shape ', x_train.shape)
print('x_test shape ', x_test.shape)
print('x_val shape ', x_val.shape)

y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)
y_val = np_utils.to_categorical(y_val, 3)

print('y_train shape ', y_train.shape)
print('y_test shape ', y_test.shape)
print('y_val shape ', y_val.shape)

saveAndCloseData()
checkingData()
checkingValData()


#--------------------------------------------------------------------


# Define the Keras TensorBoard callback.
NAME = "Fonts-Rec-512-256-128-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME),histogram_freq=1)

x_train = pickle.load(open("x_train.pickle", "rb"))
x_test = pickle.load(open("x_test.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))


model = Sequential()
model.add(Convolution2D(512,(3,3), input_shape =(28,28,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(GaussianNoise(0.1))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256,(3,3)))
model.add(BatchNormalization())
model.add(GaussianNoise(0.15))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.15))

model.add(Convolution2D(128,(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.4))
model.add(Dense(64))

model.add(Dense(3))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer =Adam(lr=0.0001), metrics = ["accuracy"])

model.summary()

history = model.fit(x_train, y_train, batch_size = 64, epochs = 54,  validation_data=(x_test, y_test) ,  callbacks=[tensorboard_callback]) 

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Save my model as h5 file
model.save('FontRecModel.h5')

