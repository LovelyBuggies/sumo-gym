import gym
import sumo_gym
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict
import pytest
from sumo_gym.envs.vrp import VRP
from sumo_gym.envs.vrp import CVRP

vertex_num: int = 5
depot_num: int = 1
edge_num: int = 5
vehicle_num: int = 3
coordinates = np.asarray([(0., 0.), (1., 0.), (2., 1.), (3., 2.), (1., 4.)])
vertices: Dict[int, npt.NDArray[Tuple[np.float64]]] = {x: y for x, y in enumerate(coordinates)}
demand: Dict[int, np.float64] = {x: y for x, y in enumerate(np.ones(vertex_num) * 10.)}
edges: npt.NDArray[Tuple[int]] = np.asarray([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
departures: Dict[int, int] = {x: y for x, y in enumerate(np.zeros(vehicle_num).astype(int))}
capacity: Dict[int, npt.NDArray[np.float64]] = {x: y for x, y in enumerate(np.ones(vertex_num) * 5)}


def test_env_init():
    # init by default
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
