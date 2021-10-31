import numpy as np
import random
import gym
import sumo_gym
from sumo_gym.envs.vrp import VRP
from sumo_gym.utils.svg_uitls import vehicle_marker
import matplotlib
import matplotlib.pyplot as plt

n_vertex = 5
n_depot = 1
n_edge = 16
n_vehicle = 3
vertices = np.asarray([(0.5, 0.5), (2.0, 1.0), (2.5, 1.5), (0.5, 2.0), (-1.5, 1.25)])
# demand = np.asarray([0.] * n_depot + [5.] * (n_vertex - n_depot))
demand = np.asarray(
    [0.0] * n_depot + (50 * np.random.random(n_vertex - n_depot)).tolist()
)
edges = np.asarray(
    [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (0, 3),
        (3, 0),
        (0, 4),
        (4, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    ]
)
departures = np.zeros(n_vehicle).astype(int)
capacity = np.ones(n_vertex) * 20

get_colors = lambda n: list(
    map(lambda i: "#" + "%06x" % random.randint(0x000000, 0x666666), range(n))
)
plot_kwargs = {
    "vrp_depot_s": 200,
    "vrp_vertex_s": 200,
    "vrp_depot_c": "darkgreen",
    "vrp_vertex_c": "navy",
    "vrp_depot_marker": r"$\otimes$",
    "vrp_vertex_marker": r"$\odot$",
    "demand_width": 0.4,
    "demand_color": get_colors(n_vertex),
    "loading_width": 0.6,
    "loading_color": get_colors(n_vehicle),
    "location_marker": vehicle_marker,
    "location_s": 2000,
    "location_c": "lightgrey",
}

env = gym.make(
    "VRP-v0",
    n_vertex=n_vertex,
    n_depot=n_depot,
    n_edge=n_edge,
    n_vehicle=n_vehicle,
    vertices=vertices,
    demand=demand,
    edges=edges,
    departures=departures,
    capacity=capacity,
)
env.plot(**plot_kwargs)
plt.savefig("./img/env_init.pdf")

for i_episode in range(3):
    observation = env.reset()
    for t in range(10):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(info)
        env.plot(**plot_kwargs)
        plt.savefig(f"./img/env_{i_episode}_{t}.pdf")
        if done:
            print("Episode finished after {} timesteps.\n".format(t + 1))
            break

env.close()
