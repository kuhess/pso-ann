# -*- coding: utf-8 -*-

import numpy as np
import numpy.random


class ParticleSwarm:
    def __init__(self, cost_func, x_low, x_high, size=50):
        self.cost_func = cost_func
        self.x_low = np.array(x_low)
        self.x_high = np.array(x_high)
        self.v_high = (self.x_high - self.x_low) / 2.
        self.v_low = -self.v_high
        self.dim = len(x_low)
        self.size = size

        self.X = np.random.uniform(self.x_low, self.x_high, (self.size, self.dim))
        self.V = np.random.uniform(self.v_low, self.v_high, (self.size, self.dim))

        self.P = self.X.copy()
        self.S = self.cost_func(self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()

    def optimize(self, epsilon=1e-3, max_iter=100):
        iteration = 0
        while self.best_score > epsilon and iteration < max_iter:
            self.update()
            iteration = iteration + 1
        return self.g

    def update(self, omega=1.0, phi_p=2.0, phi_g=2.0):
        # Velocities update
        R_p = np.random.uniform(size=(self.size, self.dim))
        R_g = np.random.uniform(size=(self.size, self.dim))

        self.V = omega * self.V \
                + phi_p * R_p * (self.P - self.X) \
                + phi_g * R_g * (self.g - self.X)

        # Velocities bounding
        self.V = np.where(self.V < self.v_low, self.v_low, self.V)
        self.V = np.where(self.V > self.v_high, self.v_high, self.V)

        # Positions update
        self.X = self.X + self.V

        # Positions bounding
        self.X = np.where(self.X < self.x_low, self.x_low, self.X)
        self.X = np.where(self.X > self.x_high, self.x_high, self.X)

        # Best scores
        scores = self.cost_func(self.X)

        better_scores_idx = scores < self.S
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()

