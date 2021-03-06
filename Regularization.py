#import the necessary packages for rehulariztion
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

#gettimg the list of images paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

#initialize the image preprocessor, loading the dataset from disk
#and reshape the data matrix and all images to 32x32
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

#encoding the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#spliting the data for training and testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#applying diffrent type of regularization
#looping over our set of regularizers
for r in (None, "l1", "l2"):
    #traing SGD callsifier using softmax loss function and the
    #specified regularization function for 10 epoch
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=10, 
        learning_rate="constant", tol=1e-3, eta0=0.01, random_state=12)
    model.fit(trainX, trainY)
    
    #evaluating the classifier
    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc*100))
    