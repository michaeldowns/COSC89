# Contains functionality for visualizing learned
# parameters
import numpy as np

import cPickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# tiles a weight matrix according to parameters
# each row of each matrix represents the inputs into one neuron in
# the output layer
def tile_image(W, image_dim, tile_x, tile_y):
    # scale all values to be between 0 and 1
    W -= W.min()
    W *= 1.0/(W.max() + 1e-8)
    
    k = 0
    tiling = np.zeros((image_dim*tile_x, image_dim*tile_y))
    for i in range(tile_x):
        for j in range(tile_y):
            w = np.reshape(W[k], (image_dim, image_dim))
            tiling[i*image_dim:(i+1)*image_dim, j*image_dim:(j+1)*image_dim] = w
            k += 1

    return tiling

# given a weight matrix or an array of frames prints, visualizes
# the final weights or the animation corresponding to the frames
def visualize_weights(W, image_dim, tile_x, tile_y, epochs, interval,
                      fig = plt.figure(), frames=[]):
    # animate image
    if len(frames) > 0:
        ani = animation.ArtistAnimation(fig, frames, interval=interval,
                                        repeat_delay=interval*epochs)
    else:
        # get output matrix
        tiling = tile_image(W, image_dim, tile_x, tile_y)
            
        plt.imshow(tiling, cmap=cm.Greys_r)

    plt.show()
