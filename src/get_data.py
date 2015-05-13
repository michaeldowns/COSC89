# Provides functionality for 1. downloading the mnist data set if it not in
# the current directory and 2. unzipping, unpickling, preprocessing, and
# storing the data in numpy array variables for later use. Various code
# snippets taken from http://deeplearning.net/tutorial/logreg.html

import cPickle
import gzip
import os

import numpy

import theano
import theano.tensor as T

import matplotlib.pyplot as plt
import matplotlib.cm as cm

filename = 'mnist.pkl.gz'

# Downloads data set to current directory if it does not already exist
def download_data():
    if not os.path.isfile(filename):
        print filename + " does not currently exist, downloading..."
        
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, filename)
    else:
        print filename + " already exists..."

# Extracts and loads the data into tuples in a numpy array
def load_data(display_image = -1):
    if not os.path.isfile(filename):
        print filename + " does not exist, cannot load..."
    else:
        print filename + " exists, loading into variables..."
        f = gzip.open(filename, 'rb')

        # each of these is a tuple consisting of 1. a matrix
        # where each row is a 784 dimensional data point and 2. a vector
        # consisting of the corresponding class labels
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        if display_image > -1 and display_image < 784:
            x = train_set[0][display_image]
            x = numpy.reshape(x, (28, 28))
            plt.imshow(x, cmap = cm.Greys_r)
            plt.show()
        
        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables
            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                dtype=theano.config.floatX),
                                  borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        
        return rval
    
    
    

if __name__ == '__main__':
    download_data()
    datasets = load_data(7)

