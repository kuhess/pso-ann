import gym
from gym.spaces.utils import flatdim
import functools
import numpy as np

from psoann.ann import MultiLayerPerceptron, MultiLayerPerceptronWeights
from psoann.pso import ParticleSwarm


env_name = "Pendulum-v1"

env = gym.make(env_name)

num_particles = 30
num_episodes = 200
num_swarm_iterations = 50


def run_ann(ann_weights: MultiLayerPerceptronWeights, obs):
    res = MultiLayerPerceptron.run(ann_weights, obs)
    output = 2 * ((res * 2) - 1)
    return output


def compute_fitness_function(
    ann_weights: MultiLayerPerceptronWeights, env: gym.Env, n_episodes: int
):
    env.reset()
    # env.state = np.random.uniform(low=(-np.pi, 0.0), high=(-np.pi, 0.0))
    env.state = np.asarray([-np.pi, 0.0])
    theta, thetadot = env.state
    obs = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    # obs = env._get_obs()
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
num_outputs = flatdim(env.action_space)
shape = [num_inputs, 10, 5, num_outputs]


cost_func = functools.partial(compute_batch_fitness, env=env, n_episodes=num_episodes)


swarm = ParticleSwarm(
    cost_func,
    num_dimensions=MultiLayerPerceptronWeights.num_dimensions(shape),
    num_particles=num_particles,
    boundaries=(-1, 1),
    # chi=0.75,
    # phi_g=4,
    # phi_p=0.05,
    # chi=0.72984,
    # phi_p=2.05,
    # phi_g=2.05,
    omega=-0.2089,
    phi_p=-0.0787,
    phi_g=3.7637,
)
print("ANN shape:", shape)
print("Dimensions:", swarm.num_dimensions)

# Train...
print(f"Initial score={swarm.best_score}")
for i in range(num_swarm_iterations):
    swarm._update()
    print(f"Iteration #{i+1}: score={swarm.best_score}")


# Render best particle!
env = gym.make(env_name, render_mode="human")
best_ann_weights = MultiLayerPerceptronWeights.from_particle_position(swarm.g, shape)
while True:
    env.reset()
    env.state = np.asarray([-np.pi, 0.0])
    theta, thetadot = env.state
    obs = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    while True:
        action = run_ann(best_ann_weights, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

env.close()
