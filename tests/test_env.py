import gym
import sumo_gym
import numpy as np
import random
from sumo_gym.utils.svg_uitls import vehicle_marker
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


def test_env_plot():
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
    assert env.plot()

    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0x000000, 0x666666), range(n)))
    plot_kwargs = {
        "vrp_depot_s": 200, "vrp_vertex_s": 200, "vrp_depot_c": 'darkgreen', "vrp_vertex_c": 'navy', \
        "vrp_depot_marker": r'$\otimes$', "vrp_vertex_marker": r'$\odot$', "demand_width": .4, \
        "demand_color": get_colors(n_vertex), "loading_width": .6, "loading_color": get_colors(n_vehicle), \
        "location_marker": vehicle_marker, "location_s": 2000, "location_c": 'lightgrey',
    }
    assert env.plot(**plot_kwargs)