import numpy as np
import gym
import sumo_gym
from sumo_gym.envs.fmp import FMP
from sumo_gym.utils.fmp_utils import Vertex, Edge, Demand
from sumo_gym.utils.fmp_utils import ChargingStation, ElectricVehicles
import random
import sys

import matplotlib.pyplot as plt

vertices = np.asarray(
    [
        Vertex(0.0, 0.0),
        Vertex(0.0, 1.0),
        Vertex(0.0, 2.0),
        Vertex(0.0, 3.0),
        Vertex(0.0, 4.0),
        Vertex(0.0, 5.0),
        Vertex(1.0, 0.0),
        Vertex(1.0, 1.0),
        Vertex(1.0, 2.0),
        Vertex(1.0, 3.0),
        Vertex(1.0, 4.0),
        Vertex(1.0, 5.0),
        Vertex(2.0, 0.0),
        Vertex(2.0, 1.0),
        Vertex(2.0, 2.0),
        Vertex(2.0, 3.0),
        Vertex(2.0, 4.0),
        Vertex(2.0, 5.0),
        Vertex(3.0, 0.0),
        Vertex(3.0, 1.0),
        Vertex(3.0, 2.0),
        Vertex(3.0, 3.0),
        Vertex(3.0, 4.0),
        Vertex(3.0, 5.0),
        Vertex(4.0, 0.0),
        Vertex(4.0, 1.0),
        Vertex(4.0, 2.0),
        Vertex(4.0, 3.0),
        Vertex(4.0, 4.0),
        Vertex(4.0, 5.0),
        Vertex(5.0, 0.0),
        Vertex(5.0, 1.0),
        Vertex(5.0, 2.0),
        Vertex(5.0, 3.0),
        Vertex(5.0, 4.0),
        Vertex(5.0, 5.0),
    ]
)

edges = np.asarray(
    [
        Edge(0, 1),
        Edge(1, 0),
        Edge(1, 2),
        Edge(2, 1),
        Edge(2, 3),
        Edge(3, 2),
        Edge(3, 4),
        Edge(4, 3),
        Edge(4, 5),
        Edge(5, 4),
        Edge(0, 6),
        Edge(6, 0),
        Edge(1, 7),
        Edge(7, 1),
        Edge(2, 8),
        Edge(8, 2),
        Edge(3, 9),
        Edge(9, 3),
        Edge(4, 10),
        Edge(10, 4),
        Edge(5, 11),
        Edge(11, 5),
        Edge(6, 7),
        Edge(7, 6),
        Edge(7, 8),
        Edge(8, 7),
        Edge(8, 9),
        Edge(9, 8),
        Edge(9, 10),
        Edge(10, 9),
        Edge(10, 11),
        Edge(11, 10),
        Edge(6, 12),
        Edge(12, 6),
        Edge(7, 13),
        Edge(13, 7),
        Edge(8, 14),
        Edge(14, 8),
        Edge(9, 15),
        Edge(15, 9),
        Edge(10, 16),
        Edge(16, 10),
        Edge(11, 17),
        Edge(17, 11),
        Edge(12, 13),
        Edge(13, 12),
        Edge(13, 14),
        Edge(14, 13),
        Edge(14, 15),
        Edge(15, 14),
        Edge(15, 16),
        Edge(16, 15),
        Edge(16, 17),
        Edge(17, 16),
        Edge(12, 18),
        Edge(18, 12),
        Edge(13, 19),
        Edge(19, 13),
        Edge(14, 20),
        Edge(20, 14),
        Edge(15, 21),
        Edge(21, 15),
        Edge(16, 22),
        Edge(22, 16),
        Edge(17, 23),
        Edge(23, 17),
        Edge(18, 19),
        Edge(19, 18),
        Edge(19, 20),
        Edge(20, 19),
        Edge(20, 21),
        Edge(21, 20),
        Edge(21, 22),
        Edge(22, 21),
        Edge(22, 23),
        Edge(23, 22),
        Edge(18, 24),
        Edge(24, 18),
        Edge(19, 25),
        Edge(25, 19),
        Edge(20, 26),
        Edge(26, 20),
        Edge(21, 27),
        Edge(27, 21),
        Edge(22, 28),
        Edge(28, 22),
        Edge(23, 29),
        Edge(29, 23),
        Edge(24, 25),
        Edge(25, 24),
        Edge(25, 26),
        Edge(26, 25),
        Edge(26, 27),
        Edge(27, 26),
        Edge(27, 28),
        Edge(28, 27),
        Edge(28, 29),
        Edge(29, 28),
        Edge(24, 30),
        Edge(30, 24),
        Edge(25, 31),
        Edge(31, 25),
        Edge(26, 32),
        Edge(32, 26),
        Edge(27, 33),
        Edge(33, 27),
        Edge(28, 34),
        Edge(34, 28),
        Edge(29, 35),
        Edge(35, 29),
        Edge(30, 31),
        Edge(31, 30),
        Edge(31, 32),
        Edge(32, 31),
        Edge(32, 33),
        Edge(33, 32),
        Edge(33, 34),
        Edge(34, 33),
        Edge(34, 35),
        Edge(35, 34),
    ]
)

n_vertex = len(vertices)
n_edge = len(edges)
n_vehicle = 1
n_electric_vehicle = 1
n_charging_station = 3
electric_vehicles = np.asarray(
    [ElectricVehicles(i, 1, 220, 50) for i in range(n_electric_vehicle)]
)
charging_stations = np.asarray(
    [
        ChargingStation(3, 220, 20),
        ChargingStation(33, 220, 20),
        ChargingStation(22, 220, 30),
    ]
)
available_vertices = np.asarray(
    [
        v
        for v in range(35)
        if v
        not in (charging_station.location for charging_station in charging_stations)
    ]
)
departures = np.asarray([19])
demand = np.asarray(
    [
        Demand(6, 4),
        Demand(5, 16),
        Demand(13, 20),
        Demand(28, 11),
        Demand(12, 30),
        Demand(27, 5),
        Demand(1, 28),
        Demand(13, 24),
        Demand(19, 18),
        Demand(11, 32),
    ]
)

print(departures)
print(demand)
print()

env = gym.make(
    "FMP-v0",
    mode="numerical",
    n_vertex=n_vertex,
    n_edge=n_edge,
    n_vehicle=n_vehicle,
    n_electric_vehicle=n_electric_vehicle,
    n_charging_station=n_charging_station,
    vertices=vertices,
    demand=demand,
    edges=edges,
    electric_vehicles=electric_vehicles,
    departures=departures,
    charging_stations=charging_stations,
)

# plt.show()
for i_episode in range(1):
    observation = env.reset()
    for t in range(150):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps.\n".format(t + 1))
            break

env.close()
