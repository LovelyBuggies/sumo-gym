import numpy as np
import numpy.typing as npt
from typing import Dict
import sumo_gym.typing


def calculate_dist(i, j, vertices) -> float:
    l1, l2 = vertices[i], vertices[j]
    return np.sqrt(np.power(l1[0] - l2[0], 2) + np.power(l1[1] - l2[1], 2))


def get_adj_list(vertices, edges) -> sumo_gym.typing.AdjListType:
    adj = [[] for _ in range(len(vertices))]
    for e in edges:
        adj[e[0]].append(e[1])

    return np.asarray(adj, dtype=object)
