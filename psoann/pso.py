import numpy as np


def minimize_pso(cost_func, num_dimensions, num_iterations):
    num_particles = num_dimensions * 2
    swarm = ParticleSwarm(cost_func, num_dimensions, num_particles)
    return swarm.minimize(num_iterations)


class PSOResult(object):
    def __init__(self, best_particle, best_score, num_iterations):
        self.best_particle = best_particle
        self.best_score = best_score
        self.num_iterations = num_iterations


class ParticleSwarm(object):
    def __init__(
        self,
        cost_func,
        num_dimensions,
        boundaries,
        num_particles,
        chi=0.72984,
        phi_p=2.05,
        phi_g=2.05,
    ):
        self.cost_func = cost_func
        self.num_dimensions = num_dimensions
        self.boundaries = boundaries

        self.num_particles = num_particles
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g

        # Initialize the particles
        # positions
        self.X = np.random.uniform(
            low=self.boundaries[0],
            high=self.boundaries[1],
            size=(self.num_particles, self.num_dimensions),
        )
        # velocities
        self.V = np.zeros(shape=(self.num_particles, self.num_dimensions))

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
        R_p = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        R_g = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        self.V = self.chi * (
            self.V
            + self.phi_p * R_p * (self.P - self.X)
            + self.phi_g * R_g * (self.g - self.X)
        )

        # Update positions
        self.X = self.X + self.V

        # Compute scores
        scores = self.cost_func(self.X)

        # Update best positions
        better_scores_idx = scores < self.S
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()

    def minimize(self, max_iter):
        for i in range(max_iter):
            self._update()

        return PSOResult(
            best_particle=self.g, best_score=self.best_score, num_iterations=max_iter
        )
