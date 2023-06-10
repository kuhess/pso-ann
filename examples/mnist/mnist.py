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
    scores = np.asarray([])
    for w in weights:
        ann_weights = ann.MultiLayerPerceptronWeights.from_particle_position(w, shape)
        score = sklearn.metrics.log_loss(
            y, ann.MultiLayerPerceptron.run(ann_weights, X)
        )
        scores = np.append(scores, score)
    return scores


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
    cost_func,
    num_dimensions=dim_weights(shape),
    num_particles=50,
    boundaries=(-1, 1),
)
print("Dimensions:", swarm.num_dimensions)

# Train with PSO...
i_pso = 0
scores_pso = [(i_pso, swarm.best_score)]
while swarm.best_score > 1e-6 and i_pso < 200:
    swarm._update()
    i_pso += 1
    scores_pso.append((i_pso, swarm.best_score))
    print(scores_pso[-1])
j
# Test...
best_weights_pso = ann.MultiLayerPerceptronWeights.from_particle_position(
    swarm.g, shape
)
y_test_pred = np.round(ann.MultiLayerPerceptron.run(best_weights_pso, X_test.T))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))


# Train with backpropagation
alpha = 0.001

i_bp = 0
weights = ann.MultiLayerPerceptronWeights.create_random(shape, boundaries=(-1, 1))
score = sklearn.metrics.log_loss(y_true.T, ann.MultiLayerPerceptron.run(weights, X.T))
scores_bp = [(i_bp, score)]

while score > 1e-6 and i_bp < 200:
    weights = ann.MultiLayerPerceptron.backpropagate(weights, X.T, y_true.T, alpha)
    score = sklearn.metrics.log_loss(
        y_true.T, ann.MultiLayerPerceptron.run(weights, X.T)
    )
    # score = ann.MultiLayerPerceptron.evaluate(weights, X.T, y_true.T)
    i_bp += 1
    scores_bp.append((i_bp, score))
    print(scores_bp[-1])


# Test...
best_weights_bp = weights
y_test_pred = np.round(ann.MultiLayerPerceptron.run(best_weights_bp, X_test.T))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))

# print(best_weights)
