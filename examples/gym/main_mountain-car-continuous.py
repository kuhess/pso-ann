import gym
from gym.spaces.utils import flatdim
import functools
import numpy as np
from datetime import datetime
import pickle

from psoann.ann import MultiLayerPerceptron, MultiLayerPerceptronWeights
from psoann.pso import ParticleSwarm


env_name = "MountainCarContinuous-v0"

env = gym.make(env_name)

num_particles = 100
num_episodes = 200
num_swarm_iterations = 10


def run_ann(ann_weights: MultiLayerPerceptronWeights, obs):
    res = MultiLayerPerceptron.run(ann_weights, obs)
    output = (res * 2) - 1
    return output


def compute_fitness_function(
    ann_weights: MultiLayerPerceptronWeights, env: gym.Env, n_episodes: int
):
    obs, _ = env.reset()
    fitness = 0
    for _ in range(n_episodes):
        action = run_ann(ann_weights, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        fitness += reward
        if terminated:  # or truncated:
            break
    return -fitness


def compute_batch_fitness(particles_pos, env: gym.Env, n_episodes: int):
    num_particles = particles_pos.shape[0]
    scores = np.empty(shape=num_particles)
    for i in range(num_particles):
        ann_weights = MultiLayerPerceptronWeights.from_particle_position(
            particles_pos[i, :], shape
        )
        scores[i] = compute_fitness_function(ann_weights, env, n_episodes)
    return scores


num_inputs = flatdim(env.observation_space)
num_outputs = 1
shape = [num_inputs, 8, 8, num_outputs]


cost_func = functools.partial(compute_batch_fitness, env=env, n_episodes=num_episodes)


swarm = ParticleSwarm(
    cost_func,
    num_dimensions=MultiLayerPerceptronWeights.num_dimensions(shape),
    num_particles=num_particles,
    boundaries=(-10, 10),
    # chi=0.75,
    # phi_g=4,
    # phi_p=0.05,
    # chi=0.72984,
    # phi_p=2.05,
    # phi_g=2.05,
    omega=-0.05,
    phi_p=0,
    phi_g=8.92,
)
print("ANN shape:", shape)
print("Dimensions:", swarm.num_dimensions)

## Train...
# print(f"Initial score={swarm.best_score}")
# for i in range(num_swarm_iterations):
#     swarm._update()
#     print(f"Iteration #{i+1}: score={swarm.best_score}")


## save best particle
# output_filename = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_best_particle.pickle'
# print(f"Dump best particle to: {output_filename}")
# best_ann_weights = MultiLayerPerceptronWeights.from_particle_position(swarm.g, shape)
# pickle.dump(best_ann_weights, open(output_filename, "wb"))

# load
best_ann_weights = pickle.load(open("lol-24h-for-that.pickle", "rb"))

# Render best particle!
env = gym.make(env_name, render_mode="human")
while True:
    obs, _ = env.reset()
    i = 0
    score = 0
    while True:
        action = run_ann(best_ann_weights, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score -= reward
        print(i, score)
        i+=1
        if terminated:
            break

env.close()
