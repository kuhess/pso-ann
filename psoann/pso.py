import numpy as np


class PSOResult(object):
    def __init__(self, best_particle, best_score, num_iterations):
        self.best_particle = best_particle
        self.best_score = best_score
        self.num_iterations = num_iterations


class ParticleSwarm(object):
    def __init__(
        self,
        cost_func,
        num_particles,
        num_dimensions,
        boundaries=None,
        chi=0.72984,
        phi_p=2.05,
        phi_g=2.05,
        omega=None,
    ):
        self.cost_func = cost_func
        self.num_dimensions = num_dimensions
        self.boundaries = boundaries if boundaries else [0.0, 1.0]

        self.num_particles = num_particles
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.omega = omega

        # Initialize the particles
        # positions
        self.X = np.random.uniform(
            low=self.boundaries[0],
            high=self.boundaries[1],
            size=(self.num_particles, self.num_dimensions),
        )
        # velocities
        self.v_max = np.abs(self.boundaries[1] - self.boundaries[0])# * 0.05

        self.V = np.random.uniform(
            low=-self.v_max,
            high=self.v_max,
            size=(self.num_particles, self.num_dimensions),
        )

        # Best positions
        self.P = self.X.copy()
        # Scores
        self.S = self.cost_func(self.X)
        # Best particle
        self.g = self.P[self.S.argmin()]
        # Best score
        self.best_score = self.S.min()

    def _update(self):
        # Velocities update
        R_p = np.random.uniform(
            low=0.0, high=1.0, size=(self.num_particles, self.num_dimensions)
        )
        R_g = np.random.uniform(
            low=0.0, high=1.0, size=(self.num_particles, self.num_dimensions)
        )

        if self.omega is None:
            self.V = self.chi * (
                self.V
                + self.phi_p * R_p * (self.P - self.X)
                + self.phi_g * R_g * (self.g - self.X)
            )
        else:
            self.V = (
                self.omega * self.V
                + self.phi_p * R_p * (self.P - self.X)
                + self.phi_g * R_g * (self.g - self.X)
            )

        # Bound velocities
        self.V = self.V.clip(min=-self.v_max, max=self.v_max)

        # Update positions
        self.X = self.X + self.V
        # Bound positions
        self.X = self.X.clip(min=self.boundaries[0], max=self.boundaries[1])

        # Compute scores
        scores = self.cost_func(self.X)

        # Update best positions
        better_scores_idx = scores < self.S
        self.P[better_scores_idx, :] = self.X[better_scores_idx, :]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin(), :]
        self.best_score = self.S.min()

    def minimize(self, max_iter):
        for _ in range(max_iter):
            self._update()

        return PSOResult(
            best_particle=self.g, best_score=self.best_score, num_iterations=max_iter
        )
