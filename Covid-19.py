import imp
from tkinter.ttk import LabeledScale
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

#constructin the argument parser and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", 
help="path to output loss/accuracy plot")
#optional path for our output model by default will be named Covid19.model
ap.add_argument("-m","--model", type=str, default="covid19.model",
help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#intializing the initial Learning-rate, number of epoch and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

#getting the list of images in our dataset directory and initalize list of data as iamges and classes
print("[INFO] loadin images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#looping over all images paths
for imagePath in imagePaths:
    #getting the labels
    label = imagePath.split(os.path.sep)[-2]

    #loading the iamge, changing color and resize to 224x224
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    #updating data and labels list
    data.append(image)
    labels.append(label)

#converting data and labels to numpy array and normalize the data
data = np.array(data)/255.0
labels = np.array(labels)

#encoding labels to 1,0 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#spliting the data to training and testing 80%/20%
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42)

#initialixze trainging data agumentation
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

#loading VGG16 network, ensuring the head FV layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(244,224,3)))

#constructing the model that will b eplaces on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4,4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#combine the head model and the base model to our model
model = Model(inputs=baseModel.input, outputs=headModel)

#looping over all the layers in the base model and freazing them
#so they will not be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

#compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#training the head of the network
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch = len(trainX) // BS,
    validation_data = (testX, testY),
    validation_steps = len(testX) // BS,
    epochs = EPOCHS)

#making porediction on testing
print("[INFO] evaluating network")
predIdxs = model.predict(testX, batch_size=BS)

#for each image in the testing set  we need to find the index 
#of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

#show formated classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

#computing the confusion matrix and using it to derice the:
#raw accuracy, sesitivity and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0,0] + cm[1,1]) / total
sensitivity = cm[0,0] / (cm[0,0] + cm[0,1])
specificity = cm[1,1] / (cm[1,0] + cm[1,1])

#showing the confusing matrix, accuracy, sesitivity and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

#plot of the training loss and accuracy 
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_acc")
plt.title("traing loss and accuracy on Covid-19 dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")