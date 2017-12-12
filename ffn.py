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

            deltaNPlus1 = np.multiply((a[-1] - Y), self.sigmoid_prime(z[-1]))
            rateOfChange = np.dot(a[-2].T, deltaNPlus1)
            self.W[-1] -= learning_rate * rateOfChange
            
            for n in range(len(self.W)-2, -1, -1):
                deltaN = np.multiply(
                                np.dot(deltaNPlus1, self.W[n+1].T),
                                self.sigmoid_prime(z[n])
                                )

                c = np.dot(a[n].T, deltaN) # acts line n-1
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
