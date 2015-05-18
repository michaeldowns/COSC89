# Implementation of the neural network class
# Various code snippets taken from http://deeplearning.net/tutorial/mlp.html#mlp

import numpy as np
import numpy.random

import theano
import theano.tensor as T

def get_activation(activation):
    res = T.tanh
    
    if activation == "sigmoid":
        res = T.nnet.sigmoid
    elif activation == "tanh":
        res = T.tanh
    elif activation == "softplus":
        res = T.nnet.softplus
    elif activation == "relu":
        res = lambda x: T.switch(x<0, 0, x)
    else:
        print "Invalid activation, using tanh"

    return res

class NeuralNetwork(object):
    """
    input: Data matrix where rows are samples
    layers: Array specifying network architecture. i.e. [784, 100, 10]
    rng: The numpy.random random generator
    activation: The nonlinearity applied to each layer
    cost: Which cost function to use
    """
    def __init__(self, input, layers, rng, activation = "relu"):
        # Initialize network parameters
        # The convention is that each column in a weight matrix
        # represents connections into one node in the output layer
        self.W = []
        self.b = []
        self.params = []
        
        for i in range(len(layers) - 1):
            bound = np.sqrt(6. / (layers[i] + layers[i+1]))
            W_values = np.asarray(
                rng.uniform(
                    low=-bound,
                    high=bound,
                    size=(layers[i], layers[i+1])
                ),
                dtype=theano.config.floatX
            )
            if activation == "sigmoid":
                W_values *= 4
                
            b_values = np.zeros((layers[i+1],), dtype=theano.config.floatX)
            
            W = theano.shared(value=W_values, name='W_{0}{1}'.format(i+1, i+2),
                              borrow=True)
            b = theano.shared(value=b_values, name='b_{0}'.format(i+2),
                              borrow=True)

            self.W.append(W)
            self.b.append(b)

            self.params.append(W)
            self.params.append(b)

        activation = get_activation(activation)

        # symbolic matrix of class membership probabilities where the number
        # of rows is the number of samples and the number of columns is the
        # number of classes
        if len(layers) == 2:
            self.p_y_given_x = T.nnet.softmax(T.dot(input,
                                                    self.W[0]) + self.b[0])
        elif len(layers) > 2:
            for i in range(len(layers) - 1):
                if i == 0:
                    self.p_y_given_x = activation(T.dot(input,
                                                        self.W[0]) + self.b[0])
                elif i == len(layers) - 2:
                    self.p_y_given_x = T.nnet.softmax(T.dot(self.p_y_given_x,
                                                        self.W[i]) + self.b[i])
                else:
                    self.p_y_given_x = activation(T.dot(self.p_y_given_x,
                                                        self.W[i]) + self.b[i])

        # class predictions
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # sum of absolute values of weights
        self.L1 = 0
        for weight in self.W:
            self.L1 += abs(weight).sum()
        
        # squared sum of weights
        self.L2 = 0
        for weight in self.W:
            self.L2 += (weight ** 2).sum()


    # learning objective is minimizing the negative log likelihood
    # which is the same as the categorical cross entropy
    def cost_function(self, y):
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
    
    # returns the percentage of incorrect predictions for the batch
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

