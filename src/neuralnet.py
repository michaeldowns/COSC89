# Implementation of the neural network class
# Various code snippets taken from http://deeplearning.net/tutorial/mlp.html#mlp

import numpy as np
import numpy.random

import theano
import theano.tensor as T

class NeuralNetwork(object):
    """
    input: Data matrix where rows are samples
    layers: Array specifying network architecture. i.e. [784, 100, 10]
    rng: The numpy.random random generator
    activation: The nonlinearity applied to each layer
    cost: Which cost function to use
    """
    def __init__(self, input, layers, rng, activation = "tanh",
                 cost = "likelihood"):
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
            if activation == theano.tensor.nnet.sigmoid:
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


        if activation == "sigmoid":
            activation = T.nnet.sigmoid
        elif activation == "tanh":
            activation = T.tanh
        else:
            print "Invalid activation, using tanh"
            activation = T.tanh
        

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

        # squared sum of weights
        self.L2 = 0
        for weight in self.W:
            self.L2 += (weight ** 2).sum()

        self.cost = cost

    # learning objective
    def cost_function(self, y):
        if self.cost == "likelihood":
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        elif self.cost == "entropy":
            return T.nnet.categorical_crossentropy(self.p_y_given_x, y).sum()
        else:
            print "Invalid cost function, using likelihood"
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    # returns the percentage of incorrect predictions for the batch
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

