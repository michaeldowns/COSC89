# Implementation of the neural network class
# Various code snippets taken from http://deeplearning.net/tutorial/mlp.html#mlp

import numpy as np
import numpy.random

import theano
import theano.tensor as T

from autoencoder import *

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
    def __init__(self, input, layers, rng, activation = "relu",
                 autoencoder_type = "none", autoencoder_data = []):
        # Initialize network parameters
        # The convention is that each column in a weight matrix
        # represents connections into one node in the output layer
        self.W = []
        self.b = []
        self.params = []

        # initialize weights randomly if we're not using an autoencoder
        if autoencoder_type == "none":
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
        else:
            # we'll assume that the network has at least two layers
            train_x = autoencoder_data[0]
            valid_x = autoencoder_data[8]
            test_x = autoencoder_data[9]
            
            BATCH_SIZE = autoencoder_data[1]
            LEARN_RATE = autoencoder_data[2]
            EPOCHS = autoencoder_data[3]

            AE_TIED = autoencoder_data[4]
            AE_TYPE_PARAMS = autoencoder_data[5]
            AE_ACTIVATION = autoencoder_data[6]
            AE_COST = autoencoder_data[7]

            ae_act = get_activation(AE_ACTIVATION)
            
            n_train_batches = train_x.get_value(borrow=True).shape[0] / BATCH_SIZE
            
            for i in range(len(layers) - 1):
                if i == 0:
                    index = T.lscalar()
                    x = T.matrix('x')
                    
                    ae = Autoencoder(x,
                                     layers[i],
                                     layers[i+1],
                                     rng,
                                     AE_TIED,
                                     autoencoder_type,
                                     AE_TYPE_PARAMS,
                                     AE_ACTIVATION,
                                     AE_COST
                                 )

                    cost = ae.cost_function(True)
                    cost_validtest = ae.cost_function(False)

                    # function that returns error on validation set
                    test_error = theano.function(
                        inputs=[],
                        outputs=cost_validtest,
                        givens={
                            x: test_x
                        }
                    )

                    # function that returns error on test set
                    validation_error = theano.function(
                        inputs=[],
                        outputs=cost_validtest,
                        givens={
                            x: valid_x
                        }
                    )
                    
                    gparams = [T.grad(cost, param) for param in ae.params]

                    updates = [
                        (param, param - LEARN_RATE*gparam)
                        for param, gparam in zip(ae.params, gparams)
                    ]

                    train_model = theano.function(
                        inputs=[index],
                        outputs=cost,
                        updates=updates,
                        givens={
                            x: train_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
                        }
                    )

                    print "Pretraining weights between layers 1 and 2..."
                    indices = range(n_train_batches)
                    
                    for epoch in range(EPOCHS):
                        print "Validation cost at start of epoch " + str(epoch+1) + ":",
                        print str(validation_error())
                        
                        rng.shuffle(indices)

                        for i in indices:
                            train_model(i)

                    print "Final test cost: " + str(test_error())

                    if autoencoder_type == "restrictive":
                        W = theano.shared(value=ae.W[0].eval(), borrow=True)
                        b = theano.shared(value=ae.b[0].eval(), borrow=True)
                    else:
                        W = theano.shared(value=ae.W[0].get_value(), borrow=True)
                        b = theano.shared(value=ae.b[0].get_value(), borrow=True)
                    
                    self.W.append(W)
                    self.b.append(b)
                    
                    self.params.append(W)
                    self.params.append(b)
                elif i == len(layers) - 2:
                    train = ae_act(T.dot(train_x, self.W[i-1]) + self.b[i-1])
                    train_x = theano.shared(value=train.eval(), borrow=True)

                    train_y = autoencoder_data[10]

                    valid = ae_act(T.dot(valid_x, self.W[i-1]) + self.b[i-1])
                    valid_x = theano.shared(value=valid.eval(), borrow=True)

                    valid_y = autoencoder_data[11]

                    test = ae_act(T.dot(test_x, self.W[i-1]) + self.b[i-1])
                    test_x = theano.shared(value=test.eval(), borrow=True)

                    test_y = autoencoder_data[12]

                    index = T.lscalar()
                    x = T.matrix('x')
                    y = T.ivector('y')

                    bound = np.sqrt(6. / (layers[i] + layers[i+1]))
                    W_values = np.asarray(
                        rng.uniform(
                            low=-bound,
                            high=bound,
                            size=(layers[i], layers[i+1])
                        ),
                        dtype=theano.config.floatX
                    )
                    if ae_act == "sigmoid":
                        W_values *= 4
                    
                    b_values = np.zeros((layers[i+1],), dtype=theano.config.floatX)
            
                    W = theano.shared(value=W_values, name='W_{0}{1}'.format(i+1, i+2),
                                      borrow=True)
                    b = theano.shared(value=b_values, name='b_{0}'.format(i+2),
                                      borrow=True)
                    
                    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)

                    cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x,
                                                                  y))

                    y_pred = T.argmax(p_y_given_x, axis=1)

                    def errors(y):
                        return T.mean(T.neq(y_pred, y))

                    params = []
                    params.append(W)
                    params.append(b)
                    
                    gparams = [T.grad(cost, param) for param in params]

                    updates = [
                        (param, param - LEARN_RATE*gparam)
                        for param, gparam in zip(params, gparams)
                    ]

                    test_error = theano.function(
                        inputs=[],
                        outputs=errors(y),
                        givens={
                            x: test_x,
                            y: test_y
                        }
                    )

                    # function that returns error on test set
                    validation_error = theano.function(
                        inputs=[],
                        outputs=errors(y),
                        givens={
                            x: valid_x,
                            y: valid_y
                        }
                    )
                    

                    train_model = theano.function(
                        inputs=[index],
                        outputs=cost,
                        updates=updates,
                        givens={
                            x: train_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE],
                            y: train_y[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
                        }
                    )

                    print "Pretraining weights between layers " + str(i+1) + " and " + str(i+2)
                    indices = range(n_train_batches)
                    
                    for epoch in range(EPOCHS):
                        print "Validation error at start of epoch " + str(epoch+1) + ":",
                        print str(validation_error()*100) + "%"
                        
                        rng.shuffle(indices)

                        for i in indices:
                            train_model(i)

                    print "Final test error: " + str(test_error()*100) + "%"

                    self.W.append(W)
                    self.b.append(b)
                    
                    self.params.append(W)
                    self.params.append(b)
                    
                    
                else:
                    # transform data
                    
                    train = ae_act(T.dot(train_x, self.W[i-1]) + self.b[i-1])
                    train_x = theano.shared(value=train.eval(), borrow=True)

                    valid = ae_act(T.dot(valid_x, self.W[i-1]) + self.b[i-1])
                    valid_x = theano.shared(value=valid.eval(), borrow=True)

                    test = ae_act(T.dot(test_x, self.W[i-1]) + self.b[i-1])
                    test_x = theano.shared(value=test.eval(), borrow=True)

                    
                    index = T.lscalar()
                    x = T.matrix('x')
                    
                    ae = Autoencoder(x,
                                     layers[i],
                                     layers[i+1],
                                     rng,
                                     AE_TIED,
                                     autoencoder_type,
                                     AE_TYPE_PARAMS,
                                     AE_ACTIVATION,
                                     AE_COST
                                 )

                    cost = ae.cost_function(True)
                    cost_validtest = ae.cost_function(False)

                    # function that returns error on validation set
                    test_error = theano.function(
                        inputs=[],
                        outputs=cost_validtest,
                        givens={
                            x: test_x
                        }
                    )

                    # function that returns error on test set
                    validation_error = theano.function(
                        inputs=[],
                        outputs=cost_validtest,
                        givens={
                            x: valid_x
                        }
                    )
                    
                    gparams = [T.grad(cost, param) for param in ae.params]

                    updates = [
                        (param, param - LEARN_RATE*gparam)
                        for param, gparam in zip(ae.params, gparams)
                    ]

                    train_model = theano.function(
                        inputs=[index],
                        outputs=cost,
                        updates=updates,
                        givens={
                            x: train_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
                        }
                    )

                    print "Pretraining weights between layers  " + str(i+1) + " and " + str(i+2)
                    indices = range(n_train_batches)
                    
                    for epoch in range(EPOCHS):
                        print "Validation cost at start of epoch " + str(epoch+1) + ":",
                        print str(validation_error())
                        
                        rng.shuffle(indices)

                        for i in indices:
                            train_model(i)

                    print "Final test cost: " + str(test_error())

                    if autoencoder_type == "restrictive":
                        W = theano.shared(value=ae.W[0].eval(), borrow=True)
                        b = theano.shared(value=ae.b[0].eval(), borrow=True)
                    else:
                        W = theano.shared(value=ae.W[0].get_value(), borrow=True)
                        b = theano.shared(value=ae.b[0].get_value(), borrow=True)
                    
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

