#Import need packages for Gradient descent
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

#Computing sgmoid activation function for given imput
#Sigmoid function similar to 'S' shaped curve, we refer it as activation function
#Because it will it will refer for "off" for values under 0.5 and as "on" for above 0.5 based on input x
def sigmoid_activation(x):
    return 1.0/(1 + np.exp(-x))

#computing the derivate of sigmoid function assuming that the input 'x'
#has already been passed through the sigmoid activation function
def sigmoid_deriv(x):   
    return x * (1-x)

#the predict function applies the acctivation function and then tresholds it based on the neuron firing or not (1 or 0)
def predict(X,W):
    #taking the dot between our features and weight matrix 
    preds = sigmoid_activation(X.dot(W))
    
    #applying step function and treshold the output to binary class label
    preds[preds<=0.5] = 0
    preds[preds>0.5] = 1
    
    #reurn the predication as binary 
    return preds

#
ap = argparse.ArgumentParser()
ap.add_argument("-e","--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())
#alpha is the hyperparamter for learning rate, we can change it as we need for diffrent problem
#usualy takes a lot of time and effort to find the best value for spesific model.
#most of thetime in gradient decent we will see learning rate of 0.1 0.001 0.0001 

#now lets generate data for classification:
#generate 2 class classification problem with 1000 points
#where each data point is 2D feature vector
(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((X.shape[0],1))

#inserting column of '1' as the last entry in the feature matrix 
#this allows us to treat the bias as a trainable parameter within the waieght matrix
X = np.c_[X, np.ones((X.shape[0]))]

#spliting traning and testing data
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

#intialize our weight matrix and list of losts
print("[INFO] training...")
W = np.random.randn(X.shape[1],1)
losses = []

#going over number of epochs
for epoch in np.arange(0, args["epochs"]):
    #taking the dot between feature X and the weight matrix W
    #after that pass the value through our sigmoid activation function
    preds = sigmoid_activation(trainX.dot(W))
    
    #after we have our predication we have we need to determine the error
    #which is the diffrence between our predication and the true value
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)
    #the gradient decent update between our features and error of the sigmiud derivate of the prediction
    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)
    
    #while updating we just need to "push" the weight matrix in the negative derication
    #of the gradient hence the name "gradient decent", taking small step torwads optimal results
    W += -args["alpha"] * gradient
    
    #checking to see if an update should be displayed
    if epoch == 0 or (epoch % 5) == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()        