import numpy as np
import gym
import sumo_gym
from sumo_gym.envs.vrp import VRP

n_vertex = 5
n_depot = 1
n_edge = 16
n_vehicle = 3
vertices = np.asarray([(0., 0.), (1., 0.), (2., 1.), (3., 2.), (1., 4.)])
demand = np.asarray([0.] * n_depot + [10.] * (n_vertex - n_depot))
edges = np.asarray([(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0), \
                    (1, 2),  (2, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)])
departures = np.zeros(n_vehicle).astype(int)
capacity = np.ones(n_vertex) * 20


env = gym.make('VRP-v0',
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

for i_episode in range(3):
    observation = env.reset()
    for t in range(10):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(info)
        if done:
            print("Episode finished after {} timesteps.\n".format(t + 1))
            break

env.close()
