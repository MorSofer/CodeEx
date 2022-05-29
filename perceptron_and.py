# import the necessary packages
from pyimagesearch.nn import Perceptron
import numpy as np

#construct the or dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

#Define our preceptron and training 
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X,y, epochs=20) 

print("[INFO] testing perceptron...")
#after we trained the model we can run over the data points
for (x, target) in zip(X,y):
    #making prediction on the data point and display the result to the consule
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))