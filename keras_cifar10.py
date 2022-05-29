# USAGE
# python keras_cifar10.py --output output/keras_cifar10.png

# import the necessary packages
from xml.sax.xmlreader import InputSource
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True,help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#loading training and testing data, scale it into the rage of 0 and 1
#then reshape the matrix 
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
#convert the data type of CIFAR-10 from unsigned 8-bit integers to floating
#point, followed by scaling the data to the range [0,1].
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
#reshaping the design matrix for the training and testing data. Recall that each image in the
#CIFAR-10 dataset is represented by a 32×32×3 image.
#For example, trainX has the shape (50000, 32, 32, 3) and testX has the shape (10000,
#32, 32, 3). If we were to flatten this image into a single list of floating point values, the list
#would have a total of 32×32×3 = 3,072 total entries in it.
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#initalize the label names for the CIFAR-10 dataset
labelName = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#define the 3072-1024-512-10 model
model = Sequential()
model.add(Dense(1024, input_shape=(3072,),activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimazer=sgd, metrics=["accarcy"])
H = model.fit(trainX, trainY,validation_data=(testX, testY), epochs=100, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1)), predictions.argmax(axis=1),target_names=labelName) 

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])