# -*- coding: utf-8 -*-

import pytest
import numpy as np
import numpy.linalg
import pso

def evaluate_simple(x):
    """Simple 2D concave surface with the minimum at the origin"""
    x = np.reshape(x, (-1, 2))
    return np.sum(x**2, axis=1)

def schaffer6(x):
    """Complex 2D surface with the global minimum at the origin"""
    x = np.reshape(x, (-1, 2))
    sumOfSquares = np.sum(x**2, axis=1)
    return 0.5 + (np.sin(np.sqrt(sumOfSquares))**2 - 0.5) / (1 + 0.001 * sumOfSquares)**2

def test_simple_optimization():
    swarm = pso.ParticleSwarm(
            cost_func=evaluate_simple,
            dim=2,
            size=20
    )

    best = swarm.optimize(epsilon=1e-6, max_iter=100000)
    assert evaluate_simple(best) < 1e-6

def test_complex_optimization():
    swarm = pso.ParticleSwarm(
            cost_func=schaffer6,
            dim=2,
            size=20
    )

    best = swarm.optimize(epsilon=1e-6, max_iter=100000)
    assert schaffer6(best) < 1e-6
