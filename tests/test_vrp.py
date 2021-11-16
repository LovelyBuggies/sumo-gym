from typing import Tuple, Dict

# import pytest
import numpy as np
import numpy.typing as npt
from sumo_gym.envs.vrp import VRP


n_vertex = 5
n_depot = 1
n_edge = 8
n_vehicle = 3
vertices = np.asarray([(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 2.0), (1.0, 4.0)])
demand = np.ones(n_vertex) * 10.0
edges = np.asarray([(0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (1, 2), (2, 3), (3, 4)])
departures = np.zeros(n_vehicle).astype(int)
capacity = np.ones(n_vertex) * 5


def test_vrp_basics():
    assert VRP(
        n_vertex=n_vertex,
        n_depot=n_depot,
        n_edge=n_edge,
        n_vehicle=n_vehicle,
        vertices=vertices,
        demand=demand,
        edges=edges,
        departures=departures,
    )


def test_cvrp_basics():
    assert VRP(
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
