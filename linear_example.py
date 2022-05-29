#Our main purpose here is to undersrand how we initialize
#the weight matrix W and the bias vector b
#Later in the book we will show to train a linear model from scratch

#Our goal here is to write a Python script that will 
#correctly classify Figure 8.2 as “dog" (img of dog)
import numpy as np
import cv2

# initialize the class labels and set the seed of the pseudorandom
# number generator so we can reproduce our results
labels = ["dog", "cat", "panda"]
np.random.seed(1)
#initializes the list of target class labels for the “Animals” dataset
#while sets the pseudorandom number generator for NumPy, ensuring that 
#we can reproduce the results of this experiment.

#be *learned* by our model, but for the sake of this example,
#let's use random value
W = np.random.randn(3,3072)
b = np.random.randn(3)
#initializes the weight matrix W with random values from a normal distribution,
#with zero mean and unit variance
#The size of the metrix is 3 for each label and 3072 for each image which is 32X32X3

#after we created the weight matrix and bias vector we can load, resize and flatten the image
# load our example image, resize it, and then flatten it into our
# "feature vector" representation
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32,32)).flatten()

# compute the output scores by taking the dot product between the
# weight matrix and image pixels, followed by adding in the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for(lable, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(lable, score))
    
# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#finaly we can display the result:
cv2.imshow("Image", orig)
cv2.waitKey(0)