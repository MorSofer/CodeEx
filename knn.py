'''Goal:
1 - gather the animal dataset of dog cat and pandas each 1000 in total 3000
2 - split the data set, as we already know each ML need training, valadaton and testing data
3 - training the K-nn calssifier
4 - last step is the evaluation
'''

#we starting by importing the need packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

#constructing the arguments parse and parse them
#here we getting the dataset, k and jobs optinal from the user in command line
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

#getting the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

#initialize the images preprocessor, loading the dataset from disk and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

#showing some info on the memoery consumption
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1024.0)))

#encoding the lavels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#partition the into trainging and testing spliting to 75% and 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#training and evaluating a K-nn calssifier on the raw pixel intesities
print("[INFO] evaluating K-nn classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
            n_jobs=args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY, model.predict(testX),
	target_names=le.classes_))