###############################################################################
# This is a test script to explore some of Theano's features. We're going to  #
# learn parameters for a quadratic function ax^2 + bx + c via batch           #
# gradient descent by creating noisy data and then minimizing a cost function #
###############################################################################
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
a = .5
b = 2
c = 1
tol = .000000001

# create data
trX = np.linspace(-4, 0, 200)
trY = a * trX ** 2 + b * trX + c + np.random.randn(*trX.shape) * 0.33

# Create theano expression graph
X = T.vector()
Y = T.vector()
w = theano.shared(np.asarray([1., 1., 1.]))
a = theano.shared(.01)
y = w[0] * T.sqr(X)  + w[1] * X + w[2]

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
updates = [(w, w - a.get_value()*gradient), (a, a + .05)]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# set up plot for animation
fig = plt.figure()
plt.plot(trX, trY, 'ko')
plt.ylim([-2, 2])

# train until desired tolerance
old_cost = train(trX, trY)
delta = 1
frames = []
while(abs(delta) > tol):
    new_cost = train(trX, trY)
    delta = new_cost - old_cost
    old_cost = new_cost
    frames.append(plt.plot(trX, np.dot(np.column_stack((trX**2, trX, np.ones_like(trX))), w.get_value()), "r--", linewidth=2))

ani = animation.ArtistAnimation(fig, frames, interval=1, repeat=False)
plt.show()
    
print w.get_value()
