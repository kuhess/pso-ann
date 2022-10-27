from dataclasses import dataclass
import numpy as np
import scipy.special


@dataclass
class MultiLayerPerceptronWeights:
    shape: list[int]
    weights: list[float]

    def num_layers(self):
        return len(self.shape)

    def num_inputs(self):
        return self.shape[0]

    def num_outputs(self):
        return self.shape[-1]

    @classmethod
    def create_random(cls, shape: list[int]):
        weights = []
        for i in range(len(shape) - 1):
            W = np.random.uniform(size=(shape[i + 1], shape[i] + 1))
            weights.append(W)
        return cls(shape, weights)


class MultiLayerPerceptron:
    @staticmethod
    def run(weights: MultiLayerPerceptronWeights, inputs):
        layer = inputs  # todo check shape => it must be a vertical vector
        for i in range(weights.num_layers() - 1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(weights.weights[i], prev_layer)
            # activation function: logistic sigmoid ]0;1[
            layer = scipy.special.expit(o)
        return layer
