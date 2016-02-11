import scipy
import random
import math
import numpy as np
import random as r
from matplotlib import pyplot as plt

# Calculate euclidean distance between prototype vector and test vector
# x1 y1 x2 y2 x3 y3 x4 y4
# a1 b1 a2 b2 a3 b3 a4 b4

LAM = 0.0001  # 'lambda' is a reserved keyword in python
SIGMA = 1 # 100 gets good results
PI = math.pi

class Wireframe(object):
    def __init__(self, vertices):
        """
        self.tl, self.tr, self.bl, self.br = self.vertices
        """

        self.vertices = vertices

    def to_vector(self):
        return np.concatenate(self.vertices)

    def _linear_transform(self, T):
        for i, vertex in enumerate(self.vertices):
            x, y = vertex
            new_vertex = (T[0] * x + T[1] * y), (T[2] * x + T[3] * y)
            self.vertices[i] = new_vertex

    def _rotate(self, theta=None):
        if theta is None:
            # Avoid relabeling corners (by only rotating max of 90 degrees)
            theta = r.random() * (PI/2)

        T = math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)
        self._linear_transform(T)

    def _scale(self, factor=None):
        if factor is None:
            random = r.random() * 20 + 1
            factor = random if random >= 1 else 1

        T = factor, 0, 0, factor
        self._linear_transform(T)

    def _translate(self):
        translate_x, translate_y = r.random() * 10, r.random() * 10

        for i, vertex in enumerate(self.vertices):
            x, y = vertex
            new_vertex = x + translate_x, y + translate_y
            self.vertices[i] = new_vertex

    def plot(self, color='black'):
        order = [0, 1, 3, 2, 0]
        xs = [self.vertices[i][0] for i in order]
        ys = [self.vertices[i][1] for i in order]

        plt.plot(xs, ys, marker='.', linestyle='solid', c=color)
        plt.ylim([-40,40])
        plt.xlim([-40,40])

    def draw(self):
        self.plot()
        plt.show()


class Square(Wireframe):
    @classmethod
    def random(cls):
        vertices = [(0, 1), (1, 1), (0, 0), (1, 0)]

        s = cls(vertices)

        s._scale()
        s._rotate()
        s._translate()

        return s

class Noise(Wireframe):
    @classmethod
    def random(cls):
        vertices = [tuple([random.randint(0, 40) for _ in range(2)]) for _ in range(4)]
        return cls(vertices)

class RBFN(object):
    """
    This takes the training data in the init, and creates an RBF Network with
    as many units as there are training examples. To get the output for a new
    input, call .test()
    """

    def __init__(self, training_data):
        # Takes list of wireframe
        self.centers = training_data
        G = self._compute_G(self.centers)
        K = len(self.centers)  # the amount of data points
        y = np.ones(K)  # all the training examples are positive examples
        self.weights = np.dot(
            np.dot(np.linalg.inv(np.dot(G.T, G) + LAM * np.identity(K)), G.T), y)

    def _to_vector(self, x):
        return np.array(x.to_vector())

    def _distance_sq(self, a, b):
        # this is equivalent to every member of (a - b) being squared and summed
        # OR ||a - b||^2 (the square of the euclidean distance)
        return np.dot(
            self._to_vector(a) - self._to_vector(b),
            self._to_vector(a) - self._to_vector(b)
        )

    def _activation(self, a, b):
        return np.exp((-1 * self._distance_sq(a, b)) / SIGMA)

    def _compute_G(self, X):
        return np.array([[self._activation(x, c) for c in self.centers] for x in X])

    def closest_center(self, datum):
        return min(self.centers, key=lambda x: self._distance_sq(datum, x))

    def furthest_center(self, datum):
        return max(self.centers, key=lambda x: self._distance_sq(datum, x))

    def test(self, data):
        G = self._compute_G(data)
        return np.dot(G, self.weights)

    def test_single(self, data):
        result = self.test([data])
        return result[0]

    def draw(self):
        columns = math.floor(math.sqrt(len(self.centers)))
        rows = columns + 1
        for i, c in enumerate(self.centers):
            plt.subplot(rows, columns, i + 1)
            c.plot()
        plt.show()


class Test(object):
    def __init__(self, net):
        self.net = net

    def _example(self, shape):
        shape.plot()

        closest = self.net.closest_center(shape)
        closest.plot(color='blue')

        furthest = self.net.furthest_center(shape)
        furthest.plot(color='red')

        plt.show()

    def example_square(self):
        sq = Square.random()
        self._example(sq)

    def example_noise(self):
        sq = Noise.random()
        self._example(sq)

    def average(self, n=10):
        print "Average on Squares: ", np.mean([self.net.test_single(Square.random()) for _ in range(n)])
        print "Average on Non-squares: ", np.mean([self.net.test_single(Noise.random()) for _ in range(n)])


# example usage
# square = Square()
# square.draw()

if __name__ == '__main__':
    def list_help(context):
        for cmd in sorted(context['cmds'].keys()):
            print cmd

    def _net_regen(context, n):
        net = RBFN([Square.random() for _ in range(n)])
        context['curnet'] = net
        context['curtest'] = Test(net)

    def net_regen_custom(context):
        n = int(raw_input('Training size: '))
        _net_regen(context, n)

    def net_regen(context):
        _net_regen(context, 100)

    def net_show(context):
        context['curnet'].draw()

    def net_avg(context):
        context['curtest'].average()

    def net_example_square(context):
        context['curtest'].example_square()

    def net_example_noise(context):
        context['curtest'].example_noise()

    def open_ipdb(context):
        import pdb; pdb.set_trace()

    def set_sigma(context):
        SIGMA = float(raw_input('Sigma: '))

    def set_lambda(context):
        LAM = float(raw_input('Lambda: '))

    context = {
        'curnet': None,
        'curtest': None
    }

    # generate first net
    net_regen(context)

    context['cmds'] = {
        'help': list_help,
        'net regen': net_regen,
        'net show': net_show,
        'net avg': net_avg,
        'net example square': net_example_square,
        'net example noise': net_example_noise,
        'set sigma': set_sigma,
        'set lambda': set_lambda,
        'net regen custom': net_regen_custom,
        'ipdb': open_ipdb
    }

    while True:
        cmd = raw_input('$ ')
        if cmd in context['cmds']:
            context['cmds'][cmd](context)
        else:
            print 'Command not found! (Type "help" for help)'
