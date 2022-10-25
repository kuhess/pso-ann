import matplotlib.colors as colors
import numpy as np


class SchafferN6:
    @staticmethod
    def evaluate(X):
        X = np.reshape(X, (-1, 2))
        sum_of_squares = np.sum(X**2, axis=1)
        return (
            0.5
            + (np.sin(np.sqrt(sum_of_squares)) ** 2 - 0.5)
            / (1 + 0.001 * sum_of_squares) ** 2
        )

    @staticmethod
    def name():
        return "Schaffer N.6 func"

    @staticmethod
    def boundaries():
        return (-30, 30)

    @staticmethod
    def score_boundaries():
        return (0, 0.1)

    @staticmethod
    def scale(Z):
        return None


class Easom:
    @staticmethod
    def evaluate(X):
        X = np.reshape(X, (-1, 2))
        A = np.sum((X - np.pi) ** 2, axis=1)
        return -np.cos(X[:, 0]) * np.cos(X[:, 1]) * np.exp(-A)

    @staticmethod
    def name():
        return "Easom function"

    @staticmethod
    def boundaries():
        return (-10, 10)

    @staticmethod
    def score_boundaries():
        return (-1, -0.9)

    @staticmethod
    def scale(Z):
        return None


class Beale:
    @staticmethod
    def evaluate(X):
        X = np.reshape(X, (-1, 2))
        x = X[:, 0]
        y = X[:, 1]

        return (
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * y * y) ** 2
            + (2.625 - x + x * y * y * y) ** 2
        )

    @staticmethod
    def name():
        return "Beale function"

    @staticmethod
    def boundaries():
        return (-4.5, 4.5)

    @staticmethod
    def score_boundaries():
        return (0, 0.1)

    @staticmethod
    def scale(Z):
        return colors.LogNorm(vmin=Z.min(), vmax=Z.max())


class Rastrigin:
    @staticmethod
    def evaluate(X):
        X = np.reshape(X, (-1, 2))
        A = 10
        return A * 2 + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

    @staticmethod
    def name():
        return "Rastrigin function"

    @staticmethod
    def boundaries():
        return (-5.12, 5.12)

    @staticmethod
    def score_boundaries():
        return (0, 1)

    @staticmethod
    def scale(Z):
        return None


class Dropwave:
    @staticmethod
    def evaluate(X):
        X = np.reshape(X, (-1, 2))
        sum_of_squares = np.sum(X**2, axis=1)
        return 1 - np.cos(np.sqrt(sum_of_squares)) / np.sqrt(sum_of_squares + 1)

    @staticmethod
    def name():
        return "dropwave function"

    @staticmethod
    def boundaries():
        return (-20, 20)

    @staticmethod
    def score_boundaries():
        return (0, 1)

    @staticmethod
    def scale(Z):
        return None
