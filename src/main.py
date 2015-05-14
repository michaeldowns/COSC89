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

######################
# NETWORK PARAMETERS #
######################
MODEL = [784, 500, 10]
BATCH_SIZE = 20
LEARN_RATE = 0.01
EPOCHS = 1000
WEIGHT_DECAY = .0001
SEED = 1234 # used to initialize weights


################
# EXTRACT DATA #
###############
print "Handling data..."

download_data()
data = load_data(use_shared=True)

train_x, train_y = data[0]
valid_x, valid_y = data[1]
test_x, test_y = data[2]

n_train_batches = train_x.get_value(borrow=True).shape[0] / BATCH_SIZE
n_valid_batches = valid_x.get_value(borrow=True).shape[0] / BATCH_SIZE
n_test_batches = test_x.get_value(borrow=True).shape[0] / BATCH_SIZE

###################
# CONSTRUCT MODEL #
###################
print "Constructing model..."

rng = numpy.random.RandomState(SEED)

index = T.lscalar()
x = T.matrix('x') # data matrix
y = T.ivector('y') # class labels 
                   
nn = neuralnet.NeuralNetwork(x, MODEL, rng)

cost = nn.negative_log_likelihood(y) + WEIGHT_DECAY/2 * nn.L2

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

updates = [
    (param, param - LEARN_RATE * gparam)
    for param, gparam in zip(nn.params, gparams)
]

# actual function that performs the updates
train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE - 1],
        y: train_y[index * BATCH_SIZE : (index + 1) * BATCH_SIZE - 1]
    }
)

###############
# TRAIN MODEL #
###############

print "Training model..."
start_time = time.clock()

indices = range(n_train_batches)

for epoch in range(EPOCHS):
    print "Epoch " + str(epoch)
    print "Validation error at start of epoch:"
    print str(validate_model()*100) + "%"
    
    # shuffle the indices
    rng.shuffle(indices)

    for i in indices:
        train_model(i)

end_time = time.clock()

print "Final test set error: "
print str(test_model()*100) + "%"

print "Ran for " + str((end_time - start_time)/60) + " minutes"
    


    
