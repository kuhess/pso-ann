import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import psoann.pso as pso


# Define objective function
def evaluate(X, name="easom"):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    X = np.reshape(X, (-1, 2))

    match name:
        case "shafferF6":
            # Shaffer's F6
            sum_of_squares = np.sum(X**2, axis=1)
            return (
                0.5
                + (np.sin(np.sqrt(sum_of_squares)) ** 2 - 0.5)
                / (1 + 0.001 * sum_of_squares) ** 2
            )
        case "easom":
            # Easom
            Z = np.sum((X - (np.ones_like(X) * np.pi)) ** 2, axis=1)
            return -np.cos(X[:, 0]) * np.cos(X[:, 1]) * np.exp(-Z)
        case _:
            # dropwave
            sum_of_squares = np.sum(X**2, axis=1)
            return 1 - np.cos(np.sqrt(sum_of_squares)) / np.sqrt(sum_of_squares + 1)


# Setup swarm of particles
objective_func_name = "shafferF6"
bounds = (-10, 10)
aspect_equal = True

swarm = pso.ParticleSwarm(
    cost_func=lambda x: evaluate(x, objective_func_name),
    num_dimensions=2,
    num_particles=10,
    boundaries=bounds,
)

# Setup the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

best_point = ax.plot([], [], [], "o", lw=2, c="r")[0]
swarm_points = ax.plot([], [], [], ".", lw=2, c="k")[0]
info_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes)


def init():
    # Draw the evaluation space
    bounds = swarm.boundaries
    xs = np.linspace(bounds[0], bounds[1], 100)
    ys = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(xs, ys)

    positions = np.concatenate((np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))), axis=1)
    Z = swarm.cost_func(positions).reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn, antialiased=True)
    if aspect_equal:
        ax.set_aspect("equal")

    # Init the particles info
    best_point.set_data_3d([], [], [])
    swarm_points.set_data_3d([], [], [])
    info_text.set_text("")

    return best_point, swarm_points, info_text


def animate(frame_number, swarm, best_point, swarm_points):
    # Update of the swarm
    swarm._update()

    # All particles
    swarm_points.set_data_3d(swarm.X[:, 0], swarm.X[:, 1], swarm.S)
    # Best particle
    best_point.set_data_3d(swarm.g[0], swarm.g[1], swarm.best_score)
    # Text
    info_text.set_text(
        "Iteration #%d\nBest score = %f\nBest position = (%f, %f)"
        % (frame_number, swarm.best_score, swarm.g[0], swarm.g[1])
    )

    return best_point, swarm_points, info_text


# Animation
ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    init_func=init,
    fargs=[swarm, best_point, swarm_points],
    interval=15,
    blit=True,
)
plt.show()
