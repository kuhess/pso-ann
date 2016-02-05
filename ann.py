# -*- coding: utf-8 -*-

import numpy as np

class MultiLayerPerceptron:
    def __init__(self, shape, weights=None):
        self.shape = shape
        self.num_layers = len(shape)
        if weights is None:
            self.weights = []
            for i in xrange(self.num_layers-1):
                W = np.random.uniform(size=(self.shape[i+1], self.shape[i] + 1))
                self.weights.append(W)
        else:
            self.weights = weights


    def run(self, data):
        layer = data.T
        for i in xrange(self.num_layers-1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(self.weights[i], prev_layer)
            layer = 1 / (1 + np.exp(-o))
        return layer


if __name__ == '__main__':
    import pso
    import functools
    import sklearn.metrics
    import sklearn.datasets
    import sklearn.cross_validation

    def dim_weights(shape):
        dim = 0
        for i in xrange(len(shape)-1):
            dim = dim + (shape[i] + 1) * shape[i+1]
        return dim

    def weights_to_vector(weights):
        w = np.asarray([])
        for i in xrange(len(weights)):
            v = weights[i].flatten()
            w = np.append(w, v)
        return w

    def vector_to_weights(vector, shape):
        weights = []
        idx = 0
        for i in xrange(len(shape)-1):
            r = shape[i+1]
            c = shape[i] + 1
            idx_min = idx
            idx_max = idx + r*c
            W = vector[idx_min:idx_max].reshape(r,c)
            weights.append(W)
        return weights

    def eval_neural_network(weights, shape, X, y):
        mse = np.asarray([])
        for w in weights:
            weights = vector_to_weights(w, shape)
            nn = MultiLayerPerceptron(shape, weights=weights)
            y_pred = nn.run(X)
            mse = np.append(mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred))
        return mse

    # Load MNIST digits from sklearn
    num_classes = 10
    mnist = sklearn.datasets.load_digits(num_classes)
    X, X_test, y, y_test = sklearn.cross_validation.train_test_split(mnist.data, mnist.target)

    num_inputs = X.shape[1]

    y_true = np.zeros((len(y), num_classes))
    for i in xrange(len(y)):
        y_true[i, y[i]] = 1

    y_test_true = np.zeros((len(y_test), num_classes))
    for i in xrange(len(y_test)):
        y_test_true[i, y_test[i]] = 1

    # Set up
    shape = (num_inputs, 64, 32, num_classes)

    cost_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y_true.T)

    swarm = pso.ParticleSwarm(cost_func, dim=dim_weights(shape), size=30)

    # Train...
    i = 0
    best_scores = [(i, swarm.best_score)]
    print best_scores[-1]
    while swarm.best_score > 1e-6 and i<500:
    #while i < 1:
        swarm.update()
        i = i+1
        if swarm.best_score < best_scores[-1][1]:
            best_scores.append((i, swarm.best_score))
            print best_scores[-1]

    # Test...
    best_weights = vector_to_weights(swarm.g, shape)
    best_nn = MultiLayerPerceptron(shape, weights=best_weights)
    y_test_pred = np.round(best_nn.run(X_test))
    print sklearn.metrics.classification_report(y_test_true, y_test_pred.T)
