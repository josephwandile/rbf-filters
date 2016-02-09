import pylab
import numpy as np
import scipy
import random as r
import math
from itertools import chain

# Vertices. Feeding in tuples. Poggio and Edelman paper is the one we're
# doing. Just use the corners.

# Inputs must be labeled. Top left. Top right. etc...

# Bring up limitations in presentation like needing alignment, labeled
# features...

# Append it all and take euclidean distance
# x1 y1 x2 y2 x3 y3 x4 y4
# a1 b1 a2 b2 a3 b3 a4 b4

IMAGE_EDGE_LENGTH = 200  # we expect each image to be 200px x 200px
IMAGE_FEATURE_LENGTH = IMAGE_EDGE_LENGTH ** 2
LAM = 0.0001  # 'lambda' is a reserved keyword in python
SIGMA = 1
PI = math.pi

# NOTE: PLEASE PLEASE PLEASE Make sure to confuse arrays and lists in comments

# NOTE: NOT ENTIRELY SURE
#
# My understanding thus far is that there are a series of RBF Units
# (one for each training example in our case). To test something, you take
# the euclidean distance between the image and the training image then you you
# weight all of those values (using the weight equation given in the HW) and see
# if it passes a threshold

# NOTE: UPDATED: STILL NOT ENTIRELY SURE
#
# Still not entirely sure, but starting to think this is simply l2-regularized
# linear regression, where the feature vector is the distance from the 'centers'
# (which are actually just all of the training points)

class square():

    def __init__(self, shape_type='square'):
        """
        | tl  tr |
        | bl  br |

        self.tl, self.tr, self.bl, self.br = self.vertices
        """

        # Instantiate column vectors
        self.vertices = [(0, 1), (1, 1), (0, 0), (1, 0)]
        self.vector_rep = np.matrix([[],[]])

        self._rotate()
        self._scale()
        self._translate()

        self.vector_rep = np.concatenate(self.vertices)

    def _linear_transform(self, T):

        for i, vertex in enumerate(self.vertices):

            x, y = vertex
            new_vertex = (T[0] * x + T[1] * y), (T[2] * x + T[3] * y)
            self.vertices[i] = new_vertex


    def _rotate(self, theta=None):

        if theta is None:
            theta = r.random() * PI * 2

        T = math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)
        self._linear_transform(T)


    def _scale(self, factor=None):

        if factor is None:
            factor = math.ceil(r.random() * 50)

        T = factor, 0, 0, factor

        self._linear_transform(T)


    def _translate(self):

        translate_x, translate_y = r.random() * 50, r.random() * 50

        for i, vertex in enumerate(self.vertices):

            x, y = vertex
            new_vertex = x + translate_x, y + translate_y
            self.vertices[i] = new_vertex

    def draw(self):

        # I don't want to talk about what I did here. Just go with it. My god. Bad coders unite.
        xs = [vertex[0] for vertex in self.vertices]
        ys = [vertex[1] for vertex in self.vertices]

        pylab.scatter(xs, ys, marker='.', s=5, c='black')
        pylab.show()


class RBF(object):
    """
    This takes the training data in the init, and creates an RBF Network with
    as many units as there are training examples. To get the output for a new
    input, call .test()
    """

    def __init__(self, training_data):
        # Training data will be a list of vectors (x1,y1,x2,y2,x3,y3,x4,y4)
        self.training_data = training_data
        self.centers = self._images_to_vectors(self.training_data)
        G = self._compute_G(self.training_data)
        K = len(self.centers)  # the amount of data points
        y = np.ones(K)  # all the training examples are positive examples
        self.weights = np.dot(
            np.dot(np.inv(np.dot(G.T, G) + LAM * np.identity(K)), G.T), y)

    def _images_to_vectors(images):
        return [np.flatten(x) for x in images]

    def _distance_sq(self, a, b):
        # this is equivalent to every member of (a - b) being squared and summed
        # OR ||a - b||^2 (the square of the euclidean distance)
        return np.dot(a - b, a - b)

    def _activation(self, a, b):
        return np.exp((-1 * self._distance_sq(a, b)) / SIGMA)

    def _compute_G(self, X):
        return np.array([[self._activation(x, c) for c in self.centers] for x in X])

    def test(self, data):
        vector_data = self._images_to_vectors(data)
        G = self._compute_G(vector_data)
        return np.dot(G, self.weights)

test = square()
test.draw()
