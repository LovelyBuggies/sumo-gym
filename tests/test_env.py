import gym
import sumo_gym
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict
import pytest
from sumo_gym.envs.vrp import VRP
from sumo_gym.envs.vrp import CVRP

vertex_num = 5
depot_num = 1
edge_num = 5
vehicle_num = 3
vertices = np.asarray([(0., 0.), (1., 0.), (2., 1.), (3., 2.), (1., 4.)])
demand = np.ones(vertex_num) * 10.
edges = np.asarray([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
departures = np.zeros(vehicle_num).astype(int)
capacity = np.ones(vertex_num) * 5

def test_env_init():
    # init by default
    with pytest.raises(ValueError):
        env = gym.make('VRP-v0')
        env.reset()
        env.render()
        env.close()

    # init via variables
    env = gym.make(
        'VRP-v0',
        vertex_num=vertex_num,
        depot_num=depot_num,
        edge_num=edge_num,
        vehicle_num=vehicle_num,
        vertices=vertices,
        demand=demand,
        edges=edges,
        departures=departures,
    )
    env.reset()
    env.render()
    env.close()

    # init via files
