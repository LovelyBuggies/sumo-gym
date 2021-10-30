import gym
import sumo_gym
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict
import pytest
from sumo_gym.envs.vrp import VRP

n_vertex = 5
n_depot = 1
n_edge = 8
n_vehicle = 3
vertices = np.asarray([(0., 0.), (1., 0.), (2., 1.), (3., 2.), (1., 4.)])
demand = np.ones(n_vertex) * 10.
edges = np.asarray([(0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (1, 2), (2, 3), (3, 4)])
departures = np.zeros(n_vehicle).astype(int)
capacity = np.ones(n_vertex) * 5

def test_env_init():
    # init by default
    with pytest.raises(ValueError):
        env = gym.make('VRP-v0')
        env.reset()
        env.close()

    # init via variables
    env = gym.make(
        'VRP-v0',
        n_vertex=n_vertex,
        n_depot=n_depot,
        n_edge=n_edge,
        n_vehicle=n_vehicle,
        vertices=vertices,
        demand=demand,
        edges=edges,
        departures=departures,
    )
    env.reset()
    env.close()

    # init via files
