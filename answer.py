from matplotlib import pyplot as plt
import numpy as np
import scipy


IMAGE_EDGE_LENGTH = 200 # we expect each image to be 200px x 200px
IMAGE_FEATURE_LENGTH = IMAGE_EDGE_LENGTH ** 2
LAM = 0.0001 # 'lambda' is a reserved keyword in python
SIGMA = 1

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


class RBF(object):
    """
    This takes the training data in the init, and creates an RBF Network with
    as many units as there are training examples. To get the output for a new
    input, call .test()
    """

    def __init__(self, training_data):
        # we expect training_data to be a list of images (which are 200x200 arrays)
        self.training_data = training_data
        self.centers = self._images_to_vectors(self.training_data)
        G = self._compute_G(self.training_data)
        K = len(self.centers) # the amount of data points
        y = np.ones(K) # all the training examples are positive examples
        self.weights = np.dot(np.dot(np.inv(np.dot(G.T, G) + LAM * np.identity(K)), G.T), y)

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


if __name__ = "__main__":
    pass
