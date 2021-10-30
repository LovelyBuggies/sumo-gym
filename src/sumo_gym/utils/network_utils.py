import numpy as np
import numpy.typing as npt
from typing import Dict

def calculate_dist(l1, l2) -> float:
    return np.sqrt( np.power(l1[0] - l2[0], 2) + np.power(l1[1] - l2[1], 2) )

def get_adj_list(vertice, edges) -> npt.NDArray[npt.NDArray[int]]:
    adj = [[] for _ in range(len(vertice))]
    for e in edges:
        adj[e[0]].append(e[1])

    return np.asarray(adj)