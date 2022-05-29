import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        #initlaize the weights metrics then store the
        #network architechure and learning rate
        self.W = []
        #layers is a list which represent the network for example [2,2,1]
        #means first layer of 2 inputs node, second hidden layer of 2 nodes and last layer 1 output node
        #usualy the output will be same as our labels
        self.layers = layers
        self.alpha = alpha
        #starting looping fron the index of the first layer but stoping before the 2 last layers
        for i in np.arange(0, len(layers) - 2):
            #randomaly initialize the weight matrix
            #conective the number of nodes in each respcitve layer togther
            #adding an extra node for the bias 
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
            #creating the randomly MxN weight matrix by sampling values from standart normal distrubition
            #the matrix size is MxN because we connecting each nodes in the layer to all the nodes in the next layer
            
        #the last 2 layer are spciel case where the input connections need a bias term but the output dosen't 
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
            
    #the next function is good for debuging 
    def __repr__(self):
        #constructing and returns string which represent the network architechure
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))
    
    #next we continue to our sigmoid function:
    def sigmoid(self, x):
        #computing and return the sigmoid activation function result on given input
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self,x):
        #computing the derivative of the sigmoid function asumming the x already
        #passed through the sigmoid function
        return x * (1 - x)
    
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        #instering a column of 1's as the last eantry feature matrix
        #-- this trick alllows us to treat the bias as a trainable paramter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        #looping over the desiralble number of epoches:
        for epoch in np.arange(0, epochs):
            #loop over each individual data point and train out network on it
            for (x, target) in zip(X,y):
                self.fit_partial(x, target)

                #check to see if we should display a training update
                if epoch == 0 or (epoch + 1) % displayUpdate == 0: 
                    loss = self.calculate_loss(X, y)
                    print("[INFO] epoch={}, loss={:.7f}".format(
					epoch + 1, loss))

    
    def fit_partial(self, x ,y):
        #constucting out list of output for each later
        #as our data points flows through the network
        #the first activation is a special casee -- it's just the input
        #feature vector itself
        A = [np.atleast_2d(x)]

        #looping over the layers of the network
        for layer in np.arange(0, len(self.W)):
            #feedforward the activation at the curent layer by taking the dot 
            #product between the activation and the weight matrx
            #this is call the "net input" to the curent layer			
            net = A[layer].dot(self.W[layer])

            #computing the "net output" is simply applying our 
            #nonlinear activation function to this input
            out = self.sigmoid(net)

            #once we have the net output, adding it to our list of activation
            A.append(out)
            #until now it was the forward passing, each layer computing the input with the nonlinear sigmoid activation function
            #and the passed it for the next layer 
            #now we can go for more complicated backwords:

            #Backpropagation
            #the first phase of backpropagation is to compute the diffrence
            #between our "predictions" (the final output activation in the activation list)
            #and the true target values
        error = A[-1] - y
            
            #from gere we need to apply the chain ruke and build our list of deltas D
            #the first entry in the deltas is simply the error of the output layer times the derivative 
            #of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

            #the first step in backpropgation is to compute the error in the prediction 
            #next we will start to apply the chain rule to construct the list of deltas D
            #once we underatnd the chain rule its becomes super east
            #to implmenting with a for loop. simply loop over the layer in revers order
            #ignoring the lsat two since we already taken them into account

        for layer in np.arange(len(A) -2, 0, -1):
                #the delta for the current layer is equal to the deltas 
                #of the pervious layer, followed by multiplying the delta
                #by the derivative of the nonlinear actiovation function
                #for the action of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

                #since we looping in reverse while building the delta list we need to reverse it
        D = D[::-1]

                #now we can update the weight matrix
        for layer in np.arange(0, len(self.W)):
            #updating our weight matrix by taking the dot product of the layer
            #activation with their respctive deltas then multyplying this value
            #by some small learnig raye and adding to our weight matrix -- this is
            #where actual "learning" takes place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
    def predict(self, X, addBias = True):
        #initialize the output prediction as the input featute -- this value will be (forward) pregagated
        #through the network obtian the final predication
        p = np.atleast_2d(X)

        #checking to see if bias coulmn shouldbe  added
        if addBias:
            #instering columns of 1's as the last entry in the feature matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        #looping over our layers in the network
        for layer in np.arange(0, len(self.W)):
            #computing the output prediction is as simple as thaking the dot product between
            #the current activation value 'p' and the weight matrix associated with the current layer
            #then passing this value through a nonlinear activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    def calculate_loss(self, X, targets):
        #making prediction for the input data point then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        #returning the loss
        return loss 