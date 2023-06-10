from datetime import datetime
import pickle
import functools
import numpy as np
import signal
from copy import deepcopy
import dataclasses

import psoann.pso as pso
from psoann.ann import MultiLayerPerceptronWeights

from neurofish.typing import random_position
from neurofish.fish import Neurofish
from neurofish.chips import Chips
from neurofish.aquarium import Aquarium


def print_best_particle(score):
    print(f"Best particle at iteration #{score[0]} has score: {score[1]}")


def store_dummy_particle(swarm: pso.ParticleSwarm, shape):
    output_filename = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_dummy_particle.pickle'
    print(f"Dump dummy particle to: {output_filename}")

    best_weights = MultiLayerPerceptronWeights.from_particle_position(swarm.P[0], shape)

    pickle.dump(best_weights, open(output_filename, "wb"))


def store_best_particle(swarm: pso.ParticleSwarm, shape):
    output_filename = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_best_particle.pickle'
    print(f"Dump best particle to: {output_filename}")

    best_weights = MultiLayerPerceptronWeights.from_particle_position(swarm.g, shape)


    pickle.dump(best_weights, open(output_filename, "wb"))


def evaluate(particle_positions, shape, width, height, num_chips, vision_resolution):
    num_particles = particle_positions.shape[0]

    scores = np.empty(shape=num_particles)

    dummy_weights = MultiLayerPerceptronWeights.create_random(shape)
    common_fish = Neurofish(
        ann_weights=dummy_weights,
        position=random_position(width, height),
        angle_rad=np.random.uniform(-np.pi, np.pi),
        vision_resolution=vision_resolution,
    )
    exclude_radius = 100
    common_chips = [
        Chips.random(
            width,
            height,
            exclude_pos=common_fish.position,
            exclude_radius=exclude_radius,
        )
        for _ in range(num_chips)
    ]

    for p in range(num_particles):
        ann_weights = MultiLayerPerceptronWeights.from_particle_position(
            particle_positions[p, :], shape
        )
        fish = dataclasses.replace(common_fish, ann_weights=ann_weights)
        chips = deepcopy(common_chips)

        aquarium = Aquarium(width, height, [fish], chips)
        for _ in range(500):
            aquarium.update()
            if len(aquarium.chips) < num_chips:
                aquarium.add_chips(
                    Chips.random(
                        width,
                        height,
                        exclude_pos=common_fish.position,
                        exclude_radius=exclude_radius,
                    )
                )
        scores[p] = -fish.num_chips_eaten
    return scores


# Set up
vision_resolution = 7
num_outputs = 2
shape = (vision_resolution + 2, 5, 3, num_outputs)
width = 1000
height = 1000
num_chips = 5
cost_func = functools.partial(
    evaluate,
    shape=shape,
    width=width,
    height=height,
    num_chips=num_chips,
    vision_resolution=vision_resolution,
)
swarm = pso.ParticleSwarm(
    cost_func,
    num_dimensions=MultiLayerPerceptronWeights.num_dimensions(shape),
    num_particles=30,
    boundaries=(-5, 5),
    # omega=-0.2089,
    # phi_p=-0.0787,
    # phi_g=3.7637,
)
print("Dimensions:", swarm.num_dimensions)

store_dummy_particle(swarm, shape)

# save best particle on Ctrl+C
def signal_handler(signum, frame):
    store_best_particle(swarm, shape)
    exit(1)
signal.signal(signal.SIGINT, signal_handler)

# Train...
print_best_particle((0, swarm.best_score))
for i in range(10):
    swarm._update()
    print_best_particle((i + 1, swarm.best_score))

# save best particle after training
store_best_particle(swarm, shape)
