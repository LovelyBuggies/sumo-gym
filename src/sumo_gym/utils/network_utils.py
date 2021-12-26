import numpy as np
import numpy.typing as npt
from typing import Dict
import sumo_gym.typing


def calculate_dist(i, j, vertices) -> float:
    l1, l2 = vertices[i], vertices[j]
    try:
        return np.sqrt(np.power(l1.x - l2.x, 2) + np.power(l1.y - l2.y, 2))
    except:
        return np.sqrt(np.power(l1[0] - l2[0], 2) + np.power(l1[1] - l2[1], 2))


def get_adj_to_list(vertices, edges) -> sumo_gym.typing.AdjListType:
    try:
        adj = [[] for _ in range(len(vertices))]
        for e in edges:
            adj[e.start].append(e.end)
        return np.asarray(adj, dtype=object)
    except:
        adj = [[] for _ in range(len(vertices))]
        for e in edges:
            adj[e[0]].append(e[1])

        return np.asarray(adj, dtype=object)


def get_adj_from_list(vertices, edges) -> sumo_gym.typing.AdjListType:
    try:
        adj = [[] for _ in range(len(vertices))]
        for e in edges:
            adj[e.end].append(e.start)
        return np.asarray(adj, dtype=object)
    except:
        adj = [[] for _ in range(len(vertices))]
        for e in edges:
            adj[e[1]].append(e[0])

        return np.asarray(adj, dtype=object)
