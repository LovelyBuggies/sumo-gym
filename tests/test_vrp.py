from typing import Tuple, Dict
# import pytest
import numpy as np
import numpy.typing as npt
from sumo_gym.envs.vrp import VRP


vertex_num = 5
depot_num = 1
edge_num = 7
vehicle_num = 3
vertices = np.asarray([(0., 0.), (1., 0.), (2., 1.), (3., 2.), (1., 4.)])
demand = np.ones(vertex_num) * 10.
edges = np.asarray([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4)])
departures = np.zeros(vehicle_num).astype(int)
capacity = np.ones(vertex_num) * 5

def test_vrp_basics():
    assert VRP(vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures)


def test_cvrp_basics():
    assert VRP(vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures, capacity)