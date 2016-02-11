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
