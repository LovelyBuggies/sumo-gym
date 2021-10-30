import numpy as np
import numpy.typing as npt
import gym
from typing import Tuple

import sumo_gym.typing

class NetworkSpace(gym.spaces.Space):
    def __init__(
            self,
            locations: sumo_gym.typing.LocationsType,
            adj_list: sumo_gym.typing.AdjListType,
            demand: sumo_gym.typing.DemandType,
            fully_loaded: npt.NDArray[bool],
            depots: npt.NDArray[int],
            shape=None,
            dtype=None,
            seed=None
    ):
        super(NetworkSpace, self).__init__()
        self.locations = locations
        self.adj_list = adj_list
        self.demand =demand
        self.fully_loaded = fully_loaded
        self.depots = depots

    def sample(self) -> npt.NDArray[int]:
        samples = np.zeros((len(self.locations)))
        for i, loc in enumerate(self.locations):
            if self.fully_loaded[i]:
                samples[i] = np.random.choice(self.depots)
            else:
                destination = []
                for v in self.adj_list[loc]:
                    if self.demand[v]:
                        destination.append(v)
                samples[i] = np.random.choice(destination, 1)[0]

        return samples


        