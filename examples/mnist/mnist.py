import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

import psoann.pso as pso
import psoann.ann as ann


def dim_weights(shape):
    dim = 0
    for i in range(len(shape) - 1):
        dim = dim + (shape[i] + 1) * shape[i + 1]
    return dim


def eval_neural_network(weights, shape, X, y):
    mse = np.asarray([])
    for w in weights:
        ann_weights = ann.MultiLayerPerceptronWeights.from_particle_position(w, shape)
        y_pred = ann.MultiLayerPerceptron.run(ann_weights, X)
        mse = np.append(
            mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred)
        )
    return mse


def print_best_particle(best_particle):
    print(
        "New best particle found at iteration #{i} with mean squared error: {score:.6f}".format(
            i=best_particle[0], score=best_particle[1]
        )
    )


# Load MNIST digits from sklearn
num_classes = 10
mnist = sklearn.datasets.load_digits(n_class=num_classes)
X, X_test, y, y_test = sklearn.model_selection.train_test_split(
    mnist.data, mnist.target
)

num_inputs = X.shape[1]

y_true = np.zeros((len(y), num_classes))
for i in range(len(y)):
    y_true[i, y[i]] = 1

y_test_true = np.zeros((len(y_test), num_classes))
for i in range(len(y_test)):
    y_test_true[i, y_test[i]] = 1

# Set up
shape = (num_inputs, 50, 30, num_classes)

cost_func = functools.partial(eval_neural_network, shape=shape, X=X.T, y=y_true.T)

swarm = pso.ParticleSwarm(
    cost_func, num_dimensions=dim_weights(shape), num_particles=30, boundaries=(-1, 1)
)
print("Dimensions:", swarm.num_dimensions)

# Train...
i = 0
best_scores = [(i, swarm.best_score)]
print_best_particle(best_scores[-1])
while swarm.best_score > 1e-6 and i < 300:
    swarm._update()
    i = i + 1
    if swarm.best_score < best_scores[-1][1]:
        best_scores.append((i, swarm.best_score))
        print_best_particle(best_scores[-1])

# Test...
best_weights = ann.MultiLayerPerceptronWeights.from_particle_position(swarm.g, shape)
y_test_pred = np.round(ann.MultiLayerPerceptron.run(best_weights, X_test.T))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))

# print(best_weights)
