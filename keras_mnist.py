# USAGE
# python keras_mnist.py --output output/keras_mnist.png

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse

#constructing the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

#loading the MNIST dataset 
print("[INFO] accessing the dataset MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

#each image in the dataset is 28x28x1, but in oreder to apply standart nn we must
#first "flatten" the image to simple list of 28x28 = 784 pixel
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28*28*1))

#scaling the data to range of 0 to 1
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

#converting the labels for integers to vectors
#right now the labels are 0 to 9 each one for each digit 
#but we want to convert it to vector of 0 and 1 which represent the right digit
#for example 1 will be [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

#define 784-256-128-10 aruchetchure in Keres
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

#training the model using SGD
print("[INFO] trainig network")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

#evulating the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

#ploting the training loss and accuracy
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