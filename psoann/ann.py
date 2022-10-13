import numpy as np
import scipy.special


class MultiLayerPerceptron:
    def __init__(self, shape, weights=None):
        self.shape = shape
        self.num_layers = len(shape)
        if weights is None:
            self.weights = []
            for i in range(self.num_layers - 1):
                W = np.random.uniform(size=(self.shape[i + 1], self.shape[i] + 1))
                self.weights.append(W)
        else:
            self.weights = weights

    def run(self, data):
        layer = data.T
        for i in range(self.num_layers - 1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(self.weights[i], prev_layer)
            # logistic sigmoid
            layer = scipy.special.expit(o)
        return layer
