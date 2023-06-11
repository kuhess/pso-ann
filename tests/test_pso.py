import numpy as np

import psoann.pso as pso


def _evaluate_simple(x):
    """Simple 2D concave surface with the minimum at the origin"""
    x = np.reshape(x, (-1, 2))
    return np.sum(x**2, axis=1)


def _schaffer6(x):
    """Complex 2D surface with the global minimum at the origin"""
    x = np.reshape(x, (-1, 2))
    sum_of_squares = np.sum(x**2, axis=1)
    return (
        0.5
        + (np.sin(np.sqrt(sum_of_squares)) ** 2 - 0.5)
        / (1 + 0.001 * sum_of_squares) ** 2
    )


def test_simple_optimization():
    swarm = pso.ParticleSwarm(
        cost_func=_evaluate_simple,
        num_dimensions=2,
        num_particles=40,
    )
    result = swarm.minimize(max_iter=100)
    assert result.best_score < 1e-6


def test_complex_optimization():
    swarm = pso.ParticleSwarm(
        cost_func=_schaffer6,
        num_dimensions=2,
        num_particles=40,
    )
    result = swarm.minimize(max_iter=100)
    assert result.best_score < 1e-6
