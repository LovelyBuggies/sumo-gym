from typing import Tuple, Dict
# import pytest
import numpy as np
import numpy.typing as npt
from sumo_gym.envs.VRP import VRP, CVRP

def test_problem_setting():
    vertex_num: int = 5
    depot_num: int = 1
    edge_num: int = 5
    vehicle_num: int = 3
    coordinates = np.asarray([(0., 0.), (1., 0.), (2., 1.), (3., 2.), (1., 4.)])
    vertices: Dict[int, npt.NDArray[Tuple[np.float64]]] = {x: y for x, y in enumerate(coordinates)}
    depots: Dict[int, npt.NDArray[Tuple[np.float64]]] = {x: y for x, y in
                                                         enumerate(list(vertices.values())[0:depot_num])}
    demand: Dict[int, np.float64] = {x: y for x, y in enumerate(np.ones(vertex_num) * 10.)}
    edges: npt.NDArray[Tuple[int]] = np.asarray([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    departures: Dict[int, int] = {x: y for x, y in enumerate(np.zeros(vehicle_num).astype(int))}
    capacity: Dict[int, npt.NDArray[np.float64]] = {x: y for x, y in enumerate(np.ones(vertex_num) * 5)}

    assert VRP(vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures)
    assert CVRP(vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures, capacity)