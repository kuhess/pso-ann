from __future__ import annotations
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
    def create_random(cls, shape: list[int]) -> MultiLayerPerceptronWeights:
        weights = []
        for i in range(len(shape) - 1):
            W = np.random.uniform(size=(shape[i + 1], shape[i] + 1))
            weights.append(W)
        return cls(shape, weights)

    @classmethod
    def from_particle_position(
        cls, particle_position: list[float], shape: list[int]
    ) -> MultiLayerPerceptronWeights:
        weights = []
        idx = 0
        for i in range(len(shape) - 1):
            r = shape[i + 1]
            c = shape[i] + 1
            idx_min = idx
            idx_max = idx + r * c
            W = particle_position[idx_min:idx_max].reshape(r, c)
            weights.append(W)
        return cls(shape, weights)

    def to_particle_position(self) -> list[float]:
        w = np.asarray([])
        for i in range(len(weights)):
            v = weights[i].flatten()
            w = np.append(w, v)
        return w


class MultiLayerPerceptron:
    @staticmethod
    def run(weights: MultiLayerPerceptronWeights, inputs):
        layer = inputs
        for i in range(weights.num_layers() - 1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(weights.weights[i], prev_layer)
            # activation function: logistic sigmoid ]0;1[
            layer = scipy.special.expit(o)
        return layer
