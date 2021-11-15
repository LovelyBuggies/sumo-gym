import numpy as np
import gym
import sumo_gym
from sumo_gym.envs.fmp import FMP
import random

import matplotlib.pyplot as plt

vertices = np.asarray([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (0.0, 5.0),
                       (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (1.0, 4.0), (1.0, 5.0),
                       (2.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0), (2.0, 4.0), (2.0, 5.0),
                       (3.0, 0.0), (3.0, 1.0), (3.0, 2.0), (3.0, 3.0), (3.0, 4.0), (3.0, 5.0),
                       (4.0, 0.0), (4.0, 1.0), (4.0, 2.0), (4.0, 3.0), (4.0, 4.0), (4.0, 5.0),
                       (5.0, 0.0), (5.0, 1.0), (5.0, 2.0), (5.0, 3.0), (5.0, 4.0), (5.0, 5.0)])

edges = np.asarray(
    [
        (0, 1),   (1, 0),   (1, 2),   (2, 1),   (2, 3),   (3, 2),   (3, 4),   (4, 3),   (4, 5),   (5, 4),
        (0, 6),   (6, 0),   (1, 7),   (7, 1),   (2, 8),   (8, 2),   (3, 9),   (9, 3),   (4, 10),  (10, 4),  (5, 11),  (11, 5),
        (6, 7),   (7, 6),   (7, 8),   (8, 7),   (8, 9),   (9, 8),   (9, 10),  (10, 9),  (10, 11), (11, 10),
        (6, 12),  (12, 6),  (7, 13),  (13, 7),  (8, 14),  (14, 8),  (9, 15),  (15, 9),  (10, 16), (16, 10), (11, 17), (17, 11),
        (12, 13), (13, 12), (13, 14), (14, 13), (14, 15), (15, 14), (15, 16), (16, 15), (16, 17), (17, 16),
        (12, 18), (18, 12), (13, 19), (19, 13), (14, 20), (20, 14), (15, 21), (21, 15), (16, 22), (22, 16), (17, 23), (23, 17),
        (18, 19), (19, 18), (19, 20), (20, 19), (20, 21), (21, 20), (21, 22), (22, 21), (22, 23), (23, 22), 
        (18, 24), (24, 18), (19, 25), (25, 19), (20, 26), (26, 20), (21, 27), (27, 21), (22, 28), (28, 22), (23, 29), (29, 23),
        (24, 25), (25, 24), (25, 26), (26, 25), (26, 27), (27, 26), (27, 28), (28, 27), (28, 29), (29, 28),
        (24, 30), (30, 24), (25, 31), (31, 25), (26, 32), (32, 26), (27, 33), (33, 27), (28, 34), (34, 28), (29, 35), (35, 29),
        (30, 31), (31, 30), (31, 32), (32, 31), (32, 33), (33, 32), (33, 34), (34, 33), (34, 35), (35, 34),
    ]
)

n_vertex = len(vertices)
n_edge = len(edges)
n_vehicle = 1
n_electric_vehicles = 1
n_charging_station = 3
electric_vehicles = np.asarray([(0, 1, 220, 50)])
charging_stations = np.asarray([(3, 220, 20), (33, 220, 20), (22, 220, 30)])
available_vertices = np.asarray([v for v in range(35) if v not in (charging_station[0] for charging_station in charging_stations)])
departures = np.asarray([19])
demand = np.asarray([(6, 4), (5, 16), (13, 20), (28, 11), (12, 30), (27, 5), (1, 28), (13, 24), (19, 18), (11, 32)])

print(departures, demand)
env = gym.make(
    "FMP-v0",
    n_vertex=n_vertex,
    n_edge=n_edge,
    n_vehicle=n_vehicle,
    n_electric_vehicles=n_electric_vehicles,
    n_charging_station=n_charging_station,
    vertices=vertices,
    demand=demand,
    edges=edges,
    electric_vehicles=electric_vehicles,
    departures=departures,
    charging_stations=charging_stations
)
env.render()
# plt.show()

for i_episode in range(3):
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps.\n".format(t + 1))
            break

env.close()
