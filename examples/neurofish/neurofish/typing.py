import numpy as np


Position = tuple([float, float])


def random_position(width, height):
    return tuple(np.random.uniform(high=[width, height], size=2))
