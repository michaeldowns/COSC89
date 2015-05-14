# Implementation of the neural network class
# Various code snippets taken from http://deeplearning.net/tutorial/mlp.html#mlp

import numpy as np
import numpy.random

import theano
import theano.tensor as T

class NeuralNetwork(object):
    """
    layers should be an array where each entry is the number of hidden units in 
    the layer i.e. [784, 50, 10]
    our neural network will have a softmax output layer by default
    and the loss function will be the negative log likelihood of the 
    data
    """
    def __init__(self, input, layers, rng, activation = T.tanh):
        # Create the network parameters
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

            W = theano.shared(value=W_values, name='W_{0}{1}'.format(i+1, i+2),
                              borrow=True)

            self.W.append(W)
            self.params.append(W)

            b_values = np.zeros((layers[i+1],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b_{0}'.format(i+2),
                              borrow=True)

            self.b.append(b)
            self.params.append(b)


        # symbolic matrix of class membership probabilities where the number
        # of rows is the number of samples and the number of columns is the
        # number of classes
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


    # determines the negative log likelihood of the model
    # with the correct class labels as inputs
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # returns the percentage of incorrect predictions for the batch
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

