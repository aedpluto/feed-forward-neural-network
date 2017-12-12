"""Created on Wed Nov 29 20:43:42 2017"""

import numpy as np
from sklearn import datasets

class NeuralNetwork(object):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.W = []
        
        for i in range(1, len(layer_sizes)):
            self.W.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]))
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def forward(self, X):
        z = []; a = [X]
        for weights in range(len(self.W)):
            z.append(np.dot(a[weights], self.W[weights]))
            a.append(self.sigmoid(z[-1]))
        return z, a
    
    def fit(self, X, Y, iterations=2000, learning_rate=0.01):
        for i in range(iterations):
            z, a = self.forward(X)
            
            # output layer
#            prevDeltan = np.multiply((a[-1] - Y), self.sigmoid_prime(z[-1]))
#            delta_weighted = np.dot(a[-2].T, prevDeltan)
#            
#            rateOfChange = learning_rate * delta_weighted
#            self.W[-1] -= rateOfChange
            
            # hidden layers
#            for n in range(len(self.W)-1, 0, -1):
#                c = np.dot(prevDeltan, self.W[n].T) # should these be the other way round? transpose?
#                deltan = np.multiply(c, self.sigmoid_prime(z[n]))
#                self.W[n] -= np.dot(deltan, a[n-1].T)
#                prevDeltan = deltan
                
            #print(len(self.layer_sizes)-2)
#            for n in range(len(self.layer_sizes)-2, 1, -1):
#                c = np.dot(prevDeltan, self.W[n].T) # following weights
#                deltan = np.multiply(c, self.sigmoid_prime(z[n]))
#                self.W[n] -= np.dot(deltan, a[n-1].T)
#                prevDeltan = deltan

            deltaNPlus1 = np.multiply((a[-1] - Y), self.sigmoid_prime(z[-1]))
            rateOfChange = np.dot(a[-2].T, deltaNPlus1)
            self.W[-1] -= learning_rate * rateOfChange
            
            for n in range(len(self.W)-2, -1, -1):
                deltaN = np.multiply(
                                np.dot(deltaNPlus1, self.W[n+1].T),
                                self.sigmoid_prime(z[n])
                                )

                #print(a[-2].shape, deltaNPlus1.shape, rateOfChange.shape)
                #print()
                #print(a[n-1].T.shape, n-1) ##### acts line n-1
                c = np.dot(a[n].T, deltaN)
                #print(deltaNPlus1.shape,
                #      self.W[n+1].T.shape,
                #      np.dot(deltaNPlus1, self.W[n+1].T).shape,
                #      self.sigmoid_prime(z[n]).shape,
                #      deltaN.shape
                #      )
                #print(c.shape, self.W[n].shape)
                #print()
                #print(n)
                self.W[n] -= learning_rate * c
                deltaNPlus1 = deltaN
            
            
    def predict(self, X):
        return self.forward(X)[1][-1]
    

class Examples(NeuralNetwork):
    def iris(iterations=25000, learning_rate=0.001, model=(4,4,4,3)):
        # preprocessing
        iris = datasets.load_iris()
        X = np.concatenate((iris.data[:48], iris.data[50:98], iris.data[100:148]), axis=0)
        test = np.concatenate((iris.data[48:50], iris.data[98:100], iris.data[148:150]), axis=0)
    
        Y = []
        options = [[1,0,0],[0,1,0],[0,0,1]]
        for c in range(3):
            for i in range(int(len(X)/3)):
                Y.append(options[c])
        Y = np.array(Y)
    
        # initialize
        NN = NeuralNetwork(model)
    
        # training
        NN.fit(X, Y, iterations, learning_rate)
    
        # testing
        hypothesis = NN.predict(test)
        for line in hypothesis:
            l = [line[0], line[1], line[2]]
            i = l.index(max(l))
            if i == 0:
                print('Setosa')
            if i == 1:
                print('Versicolor')
            if i == 2:
                print('Virginica')
Examples.iris()

# Finally got working on 12/12/2017 at around 16:52