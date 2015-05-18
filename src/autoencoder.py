# Implementation of Autoencoder class

import numpy as np
import numpy.random

import theano
import theano.tensor as T

import visualization as viz

from get_data import *

def initialize_weights(rng, input_dim, hidden_dim, activation):
    bound = np.sqrt(6. / (input_dim + hidden_dim))
    W_values = np.asarray(
        rng.uniform(
            low=-bound,
            high=bound,
            size=(input_dim, hidden_dim)
        ),
        dtype=theano.config.floatX
    )
    if activation == "sigmoid":
        W_values *= 4

    W = theano.shared(value=W_values, borrow=True)

    return W

def initialize_bias(dim):
    b_values = np.zeros((dim,), dtype=theano.config.floatX)

    b = theano.shared(value=b_values, borrow=True)

    return b

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
    
class Autoencoder(object):
    """
    input: A symbolic variable, should be a data matrix
    hidden_dim: Number of nodes in the hidden layer
    ae_type: One of normal, sparse, denoising, contractive, or restrictive
    type_params: The parameters specific to a particular type of autoencoder
        normal: []
        sparse: []
        denoising: [corruption_level]
    activation: The activation function used in the network, one of tanh, 
    sigmoid, softplus, or relu
    cost: one of entropy or quadratic
    """
    def __init__(self, input, input_dim, hidden_dim, rng, tied=False,
                 ae_type="normal", type_params=[], activation = "sigmoid",
                 cost = "entropy"):

        activation = get_activation(activation)

        self.input = input
        self.cost = cost

        self.type_params = type_params
        
        self.W = []
        self.b = []
        self.params = []
                
        # initialize the network weights. these will be initialized the same way
        # as long as the type is not restrictive
        if ae_type in ["normal", "sparse", "denoising", "contractive"]:
            # initialize weights connecting layers 1 and 2
            W_12 = initialize_weights(rng, input_dim, hidden_dim, activation)
            self.W.append(W_12)
            self.params.append(W_12)
            
            # initialize bias in layer 2
            b_2 = initialize_bias(hidden_dim)
            self.b.append(b_2)
            self.params.append(b_2)

            # if not tied initialize weights connecting layers 2 and 3
            # otherwise these weights will be the transpose of the weights
            # connecting layers 1 and 2
            if tied:
                W_23 = W_12.T
            else:
                W_23 = initialize_weights(rng, hidden_dim,
                                          input_dim, activation)
                self.params.append(W_23)

            self.W.append(W_23)
            
            # initialize bias in layer 3
            b_3 = initialize_bias(input_dim)
            self.b.append(b_3)
            self.params.append(b_3)

        elif ae_type == "restrictive":
            print "not yet coded"

        if not tied:
            self.L2 = (self.W[0]**2).sum() + (self.W[1]**2).sum()
        else:
            self.L2 = (self.W[0]).sum()

        # if type is denoising then corrupt inputs
        if ae_type == "denoising":
            corruption_level = 0.2
            if len(type_params) > 0:
                corruption_level = type_params[0]

            mask = rng.binomial(n=1, p=1- corruption_level, size=(20, 784))
            input = input * mask
            
        # output for normal autoencoder
        a = activation(T.dot(input, self.W[0]) + self.b[0])
        self.output = activation(T.dot(a, self.W[1]) + self.b[1])
        
        if ae_type == "sparse":
            print "not yet coded"
        elif ae_type == "contractive":
            print "not yet coded"
        elif ae_type == "restrictive":
            print "not yet coded"
        elif ae_type != "normal":
            print "invalid autoencoder type, using normal"

    def cost_function(self):
        cost = T.mean(T.nnet.binary_crossentropy(self.output, self.input))

        if self.cost == "quadratic":
            cost = T.mean((self.output-self.input)**2)
        elif self.cost != "entropy":
            print "Invalid cost, using entropy"

        return cost
            

if __name__ == '__main__':
    BATCH_SIZE = 20
    LEARN_RATE = 0.1
    EPOCHS = 15
    
    ################
    # EXTRACT DATA #
    ################
    print "Handling data..."

    download_data()
    data = load_data(use_shared=True)

    train_x, train_y = data[0]
    valid_x, valid_y = data[1]
    test_x, test_y = data[2]

    n_train_batches = train_x.get_value(borrow=True).shape[0] / BATCH_SIZE

    ###################
    # CONSTRUCT MODEL #
    ###################
    print "Constructing Autoencoder..."

    rng = numpy.random.RandomState(1234)
    
    index = T.lscalar()
    x = T.matrix('x') # data matrix
    
    ae = Autoencoder(x,
                     784,
                     100,
                     rng,
                     tied=True,
                     ae_type="denoising",
                     type_params=[],
                     activation="sigmoid",
                     cost="entropy"
                 )

    cost = ae.cost_function()

    # function that returns error on validation set
    test_error = theano.function(
        inputs=[],
        outputs=cost,
        givens={
            x: test_x
        }
    )

    # function that returns error on test set
    validation_error = theano.function(
        inputs=[],
        outputs=cost,
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
    
    
    ###############
    # TRAIN MODEL #
    ###############

    print "Training model..."

    indices = range(n_train_batches)

    for epoch in range(EPOCHS):
        print "Validation cost at start of epoch " + str(epoch+1) + ":",
        print str(validation_error())
    
        # shuffle the indices
        rng.shuffle(indices)

        for i in indices:
            train_model(i)


    print "Final test cost: " + str(test_error())
            
    W = np.transpose(ae.W[0].get_value(borrow=False))

    viz.visualize_weights(W, 28, 10, 10, 20, 50)
