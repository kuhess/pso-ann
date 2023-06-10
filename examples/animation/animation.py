import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import psoann.pso as pso

import optim_functions

# Optimization function
optimization_func_class = optim_functions.Rastrigin

# Path of animation output file
# if variable is set to None, the animation will not be saved
animation_file = None
# animation_file = "animation.mp4"

frame_per_sec = 25
duration = 10
num_frames = duration * frame_per_sec


# Setup swarm of particles
swarm = pso.ParticleSwarm(
    cost_func=optimization_func_class.evaluate,
    num_dimensions=2,
    num_particles=40,
    boundaries=optimization_func_class.boundaries(),
    # chi=0.98,
    # phi_g=0.1,
    # phi_p=0.1,
    omega=-0.2089,
    phi_p=-0.0787,
    phi_g=3.7637,
)

# Setup the figure
plt.rcParams.update({"font.size": 9})
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
fig = plt.figure(figsize=(960 * px, 540 * px))

fig.suptitle(
    f"Optimization of {optimization_func_class.name()} with a particle swarm",
    size=14,
    fontweight="bold",
)

spec = fig.add_gridspec(3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1])

ax_main = fig.add_subplot(spec[:, 0], aspect="equal")
ax_main.set_xlim(optimization_func_class.boundaries())
ax_main.set_ylim(optimization_func_class.boundaries())
ax_main.set_title("Top view")
ax_main.set_xlabel("x")
ax_main.set_ylabel("y")
ax_main.text(
    0.95,
    0.95,
    f"{swarm.num_particles} particles",
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax_main.transAxes,
)

ax_3d = fig.add_subplot(spec[0, 1], projection="3d")
ax_3d.set_xlim(optimization_func_class.boundaries())
ax_3d.set_ylim(optimization_func_class.boundaries())
ax_3d.set_title("3D view")
ax_3d.set_xlabel("x")
ax_3d.set_ylabel("y")
ax_3d.set_zlabel("score")
ax_3d.axes.xaxis.set_ticklabels([])
ax_3d.axes.yaxis.set_ticklabels([])
ax_3d.axes.zaxis.set_ticklabels([])

ax_text = fig.add_subplot(spec[1, 1])
ax_text.set_axis_off()

ax_score = fig.add_subplot(spec[2, 1])
ax_score.set_title("Best score")
ax_score.set_xlabel("# iterations")
ax_score.set_ylabel("score")
ax_score.set_xlim(0, num_frames)
ax_score.set_ylim(
    optimization_func_class.score_boundaries()[0],
    optimization_func_class.score_boundaries()[1],
)
scores_data = np.full(num_frames, np.nan)
scores_iter = np.arange(num_frames)
scores = ax_score.plot(scores_iter, scores_data)[0]

best_point = ax_main.plot([], [], [], marker="x", markersize=6, color="r", zorder=50)[0]
swarm_points = ax_main.scatter(
    np.zeros((swarm.num_particles, 1)),
    np.zeros((swarm.num_particles, 1)),
    marker="o",
    s=2**2,
    color="k",
    zorder=10,
)
swarm_speeds = ax_main.quiver(
    np.zeros((swarm.num_particles, 1)),
    np.zeros((swarm.num_particles, 1)),
    width=0.005,
    scale_units="xy",
    scale=1,
    zorder=100,
)
info_text = ax_text.text(
    0.05,
    0.5,
    "",
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax_text.transAxes,
)


def init():
    # Draw the evaluation space
    bounds = swarm.boundaries
    xs = np.linspace(bounds[0], bounds[1], 100)
    ys = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(xs, ys)

    positions = np.concatenate((np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))), axis=1)
    Z = swarm.cost_func(positions).reshape(X.shape)

    cmap = cm.viridis
    norm = optimization_func_class.scale(Z)
    ax_main.pcolor(X, Y, Z, cmap=cmap, zorder=1, norm=norm)

    ax_3d.plot_surface(X, Y, Z, cmap=cmap, antialiased=True, norm=norm)

    return best_point, swarm_points, swarm_speeds, info_text, scores


def animate(frame_number, best_point, swarm_points, swarm_speeds, info_text, scores):
    # Best score by iteration
    scores_data[frame_number] = swarm.best_score

    # Update of the swarm
    swarm._update()

    # All particles
    swarm_points.set_offsets(swarm.X)
    swarm_speeds.set_offsets(swarm.X)
    swarm_speeds.set_UVC(swarm.V[:, 0], swarm.V[:, 1])
    # Best particle
    best_point.set_data(swarm.g[0], swarm.g[1])

    # Best score by iteration
    scores.set_ydata(scores_data)

    # Text
    info_text.set_text(
        "\n".join(
            [
                f"""Best particle:
    x = {swarm.g[0]:.5f}
    y = {swarm.g[1]:.5f}
    score = {swarm.best_score:.5f}""",
            ]
        )
    )

    return best_point, swarm_points, swarm_speeds, info_text, scores


# Animation
ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    init_func=init,
    fargs=[best_point, swarm_points, swarm_speeds, info_text, scores],
    interval=frame_per_sec,
    blit=True,
    frames=num_frames,
)

if animation_file is None:
    plt.show()
else:
    ani.save("animation.mp4", fps=frame_per_sec, dpi=plt.rcParams["figure.dpi"] * 2)
