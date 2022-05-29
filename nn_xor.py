# USAGE
# python nn_xor.py

# import the necessary packages
from pyimagesearch.nn import NeuralNetwork
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our 2-2-1 neural network and train it
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)


#after the training we loop over xor data poings
for (x, target) in zip(X,y):
    #making prediction on the data point and dispay the result to our consule
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-true={}, pred={:.4f}, step={}".format(
        x, target, pred, step))
