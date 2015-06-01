# Implementation of Autoencoder class

import numpy as np
import numpy.random

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import visualization as viz

from get_data import *

# Given a numpy rng, dimensions, and an appropriate activation function,
# returns a symbolic theano shared variable matrix 
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
    ae_type: One of normal, denoising, contractive, or restrictive
    type_params: The parameters specific to a particular type of autoencoder
        normal: []
        denoising: [corruption_level]
        contractive: [contraction_level]
        restrictive: [alpha, which_params]
           which_params is 1, 2, or 3: 1 uses only U, 2 uses only V, 3 uses both
        sparse: [rho, sparsity]
    activation: The activation function used in the network, one of tanh, 
    sigmoid, softplus, or relu
    cost: one of entropy or quadratic
    """
    def __init__(self, input, input_dim, hidden_dim, rng, tied=True,
                 ae_type="normal", type_params=[], activation = "sigmoid",
                 cost = "entropy"):

        activation = get_activation(activation)
        self.activation = activation
        
        self.input = input
        self.cost = cost
        self.rng = rng
        
        self.type_params = type_params
        
        self.W = []
        self.b = []
        self.params = []
        
        self.ae_type = ae_type
        self.type_params = type_params

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.batchsize = input.shape[0]
        
        # initialize the network weights. these will be initialized the same way
        # as long as the type is not restrictive
        if ae_type in ["normal", "denoising", "contractive", "sparse"]:
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
            alpha = hidden_dim/2
            if len(type_params) > 0:
                alpha = type_params[0]
                
            # initialize v, which has dimensions alpha x hidden_dim
            bound = np.sqrt(6. / (input_dim + hidden_dim))
            V_values = np.asarray(
                rng.uniform(
                    low=-bound,
                    high=bound,
                    size=(alpha, hidden_dim)
                ),
                dtype=theano.config.floatX
            )
            if activation == "sigmoid":
                V_values *= 4

            V = theano.shared(value=V_values, borrow=True)
            if type_params[1] == 2 or type_params[1] == 3:
                self.params.append(V)
            
            U_values = np.asarray(
                rng.uniform(
                    low=-bound,
                    high=bound,
                    size=(input_dim, alpha)
                ),
                dtype=theano.config.floatX
            )
            if activation == "sigmoid":
                U_values *= 4

            U = theano.shared(value=U_values, borrow=True)
            if type_params[1] == 1 or type_params[1] == 3:
                self.params.append(U)
            
            # multiply them to get W_12
            W_12 = T.dot(U, V)
            self.W.append(W_12)

            b_2 = initialize_bias(hidden_dim)
            self.b.append(b_2)
            self.params.append(b_2)

            W_23 = W_12.T
            self.W.append(W_23)
            
            # initialize bias in layer 3
            b_3 = initialize_bias(input_dim)
            self.b.append(b_3)
            self.params.append(b_3)

            

        if not tied:
            self.L2 = (self.W[0]**2).sum() + (self.W[1]**2).sum()
        else:
            self.L2 = (self.W[0]).sum()

    def corrupt(self, input):
        corruption_level = 0.2
        if len(self.type_params) > 0:
            corruption_level = self.type_params[0]

        # set up theano rng
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        # corrupt inputs
        mask = theano_rng.binomial(size=input.shape, n=1,
                                   p=1 - corruption_level,
                                   dtype=theano.config.floatX)

        return mask * input
        

    # code snippet taken from deep learning tutorial
    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis
        """
        return T.reshape(hidden * (1 - hidden),
                         (self.batchsize, 1, self.hidden_dim)) * T.reshape(
                             W, (1, self.input_dim, self.hidden_dim))
    
    def cost_function(self, corrupt_inputs = False):
        if corrupt_inputs and self.ae_type == "denoising":
            input = self.corrupt(self.input)
        else:
            input = self.input
            
        # get output
        h = self.activation(T.dot(input, self.W[0]) + self.b[0])
        output = self.activation(T.dot(h, self.W[1]) + self.b[1])

        cost = T.mean(T.nnet.binary_crossentropy(output, self.input))

        if self.cost == "quadratic":
            cost = T.mean((output-self.input)**2)
        elif self.cost != "entropy":
            print "Invalid cost, using entropy"

        # add contractive cost if type is contractive
        if self.ae_type == "contractive":
            J = self.get_jacobian(h, self.W[0])
            J = T.sum(J ** 2) / 20

            contraction_level = 0.1
            if len(self.type_params) > 0:
                contraction_level = self.type_params[0]
            
            cost += contraction_level * T.mean(J)

        # add sparsity cost if type is sparse
        if self.ae_type == "sparse":
            rho = 0.05
            sparsity = 0.1

            if len(self.type_params) == 2:
                rho = self.type_params[0]
                sparsity = self.type_params[1]


            rho_hat = T.mean(h, axis=0)
            KL = rho*T.log(rho/rho_hat) + (1-rho)*T.log((1-rho)/(1-rho_hat))

            cost += sparsity * KL.mean()

        
        return cost
            

if __name__ == '__main__':
    BATCH_SIZE = 20
    LEARN_RATE = 0.1
    EPOCHS = 15
    ALPHA = 40
    SEED = 1234
    
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

    rng = numpy.random.RandomState(SEED)
    
    index = T.lscalar()
    x = T.matrix('x') # data matrix
    
    ae = Autoencoder(x,
                     784,
                     100,
                     rng,
                     tied=True,
                     ae_type="sparse",
                     type_params=[0.01, 0.1],
                     activation="sigmoid",
                     cost="entropy"
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
            
    #W = np.transpose(ae.W[0].get_value(borrow=False))
    W = np.transpose(ae.W[0].eval())

    viz.visualize_weights(W, 28, 10, 10, 20, 50)
