import numpy as np
import numpy.typing as npt
from typing import Dict
import sumo_gym.typing


def calculate_dist(i, j, vertices) -> float:
    l1, l2 = vertices[i], vertices[j]
    return np.sqrt(np.power(l1.x - l2.x, 2) + np.power(l1.y - l2.y, 2))


def get_adj_list(vertices, edges) -> sumo_gym.typing.AdjListType:
    adj = [[] for _ in range(len(vertices))]
    for e in edges:
        adj[e.start].append(e.end)

    return np.asarray(adj, dtype=object)
