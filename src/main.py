# Main script for experiments
# Code snippets taken from http://deeplearning.net/tutorial

###########
# IMPORTS #
###########
import time

import numpy as np
import numpy.random

import theano
import theano.tensor as T

from get_data import *
import neuralnet

import visualization as viz

import matplotlib.pyplot as plt

######################
# NETWORK PARAMETERS #
######################
MODEL = [784, 500, 250, 100, 10]
ACTIVATION = "sigmoid" # options are "tanh", "softplus", "relu", and "sigmoid"
BATCH_SIZE = 20
LEARN_RATE = 0.2
EPOCHS = 50
WEIGHT_DECAY = 0.0001
MOMENTUM = 0
SEED = 1234 # used to initialize weights

AE_BATCH_SIZE = 20
AE_LEARN_RATE = 0.1
AE_EPOCHS = 100
AE_TIED = True
AE_TYPE_PARAMS = []
AE_ACTIVATION = "sigmoid"
AE_COST = "entropy"
AUTOENCODER_TYPE = "normal"

VISUALIZE_WEIGHTS = False

ANIMATE_WEIGHTS = True
IMAGE_DIM = 28
TILE_X = 10
TILE_Y = 10
ANIMATION_INTERVAL = 100

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
print "Constructing model..."

rng = numpy.random.RandomState(SEED)

index = T.lscalar()
x = T.matrix('x') # data matrix
y = T.ivector('y') # class labels 

autoencoder_data = [train_x,
                    AE_BATCH_SIZE,
                    AE_LEARN_RATE,
                    AE_EPOCHS,
                    AE_TIED,
                    AE_TYPE_PARAMS,
                    AE_ACTIVATION,
                    AE_COST,
                    valid_x,
                    test_x,
                    train_y,
                    valid_y,
                    test_y
                ]

nn = neuralnet.NeuralNetwork(x, MODEL, rng, ACTIVATION, AUTOENCODER_TYPE, autoencoder_data)

cost = nn.cost_function(y) + WEIGHT_DECAY/2 * nn.L2

# function that returns error on validation set
test_model = theano.function(
    inputs=[],
    outputs=nn.errors(y),
    givens={
        x: test_x,
        y: test_y
    }
)

# function that returns error on test set
validate_model = theano.function(
    inputs=[],
    outputs=nn.errors(y),
    givens={
        x: valid_x,
        y: valid_y
    }
)

# determine the gradient and updates for each parameter
gparams = [T.grad(cost, param) for param in nn.params]

# Initialize velocities for momentum updates
V = []
for param in nn.params:
    v_values = np.zeros(tuple(param.shape.eval()), dtype=theano.config.floatX)
    v = theano.shared(value=v_values, borrow=True)
    V.append(v)
    

updates = [
    (v, MOMENTUM*v - LEARN_RATE*gparam)
    for v, gparam in zip(V, gparams)
]

m_updates = [
    (param, param + v)
    for param, v in zip(nn.params, V)
]

# actual function that performs the updates
train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE],
        y: train_y[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
    }
)

momentum_updates = theano.function(
    inputs=[],
    outputs=[],
    updates=m_updates
)

###############
# TRAIN MODEL #
###############

print "Training model..."
start_time = time.clock()

indices = range(n_train_batches)

frames = []
fig = plt.figure()
diff = 0
validerrs = []
for epoch in range(EPOCHS):
    print "Validation error at start of epoch " + str(epoch+1) + ":",
    ve = validate_model()
    validerrs.append(ve)
    print str(ve*100) + "%"
    
    # shuffle the indices
    rng.shuffle(indices)

    for i in indices:
        train_model(i)
        momentum_updates()
        
    if VISUALIZE_WEIGHTS and ANIMATE_WEIGHTS:
        W = np.transpose(nn.W[0].get_value(borrow=False))
        tiling = viz.tile_image(W, IMAGE_DIM, TILE_X, TILE_Y)
        im = plt.imshow(tiling, cmap=cm.Greys_r)
        frames.append([im])

end_time = time.clock()

print "Final test set error:",
print str(test_model()*100) + "%"

print "Ran for " + str((end_time - start_time)/60) + " minutes"

print "Network parameters are..."

print "MODEL = " + str(MODEL)
print "ACTIVATION = " + str(ACTIVATION)
print "BATCH_SIZE = " + str(BATCH_SIZE)
print "LEARN_RATE = " + str(LEARN_RATE)
print "EPOCHS = " + str(EPOCHS)
print "WEIGHT_DECAY = " + str(WEIGHT_DECAY)
print "MOMENTUM = " + str(MOMENTUM)
print "SEED = " + str(SEED)

print "AE_BATCH_SIZE = " + str(AE_BATCH_SIZE)
print "AE_LEARN_RATE = " + str(AE_LEARN_RATE)
print "AE_EPOCHS = " + str(AE_EPOCHS)
print "AE_TIED = " + str(AE_TIED)
print "AE_TYPE_PARAMS = " + str(AE_TYPE_PARAMS)
print "AE_ACTIVATION = " + str(AE_ACTIVATION)
print "AE_COST = " + str(AE_COST)
print "AUTOENCODER_TYPE = " + str(AUTOENCODER_TYPE)

if not VISUALIZE_WEIGHTS:
    print "Plotting validation errors..."
    plt.plot(validerrs, linewidth = 2.0, color = 'r')
    plt.ylabel('Validation Set Errors')
    plt.xlabel('Epoch')
    plt.ylim(0,.1)
    plt.show()

#####################
# VISUALIZE RESULTS #
#####################

if VISUALIZE_WEIGHTS:
    # plot resulting learned weights for first layer
    W = np.transpose(nn.W[0].get_value(borrow=False))

    viz.visualize_weights(W, IMAGE_DIM, TILE_X, TILE_Y,
                          EPOCHS, ANIMATION_INTERVAL, fig, frames)




    
