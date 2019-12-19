# -*- coding: utf-8 -*-

import pytest
import numpy as np

import pso


def _evaluate_simple(x):
    """Simple 2D concave surface with the minimum at the origin"""
    x = np.reshape(x, (-1, 2))
    return np.sum(x**2, axis=1)


def _schaffer6(x):
    """Complex 2D surface with the global minimum at the origin"""
    x = np.reshape(x, (-1, 2))
    sumOfSquares = np.sum(x**2, axis=1)
    return 0.5 + (np.sin(np.sqrt(sumOfSquares))**2 - 0.5) / (1 + 0.001 * sumOfSquares)**2


def test_simple_optimization():
    result = pso.minimize_pso(
            cost_func=_evaluate_simple,
            num_dimensions=2,
            num_iterations=100
    )
    assert result.best_score < 1e-6


def test_complex_optimization():
    result = pso.minimize_pso(
        cost_func=_schaffer6,
        num_dimensions=2,
        num_iterations=100
    )
    assert result.best_score < 1e-6
