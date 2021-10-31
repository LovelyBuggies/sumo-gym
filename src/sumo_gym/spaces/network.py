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
        seed=None,
    ):
        super(NetworkSpace, self).__init__()
        self.locations = locations
        self.adj_list = adj_list
        self.demand = demand
        self.fully_loaded = fully_loaded
        self.depots = depots

    def sample(self) -> npt.NDArray[int]:
        samples = np.copy(self.locations)
        for i, loc in enumerate(self.locations):
            if self.fully_loaded[i]:
                samples[i] = np.random.choice(self.depots)
            else:
                # todo: need to bypass to search the demand
                # currently, we are not bypassing, i.e., never go to the vertices that are not connected
                destination = []
                for v in self.adj_list[loc]:
                    if self.demand[v] or v in self.depots:  # if is depot or have demand
                        destination.append(v)

                if len(destination):
                    samples[i] = np.random.choice(destination, 1)[0]

        # avoid crash
        for j, s in enumerate(samples):
            if s in np.delete(samples, j) and s not in self.depots:
                samples[j] = self.locations[j]

        return samples.astype(int)
