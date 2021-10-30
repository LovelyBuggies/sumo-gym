import numpy as np
import numpy.typing as npt
import gym
from typing import Tuple

class Network(gym.spaces.Space):
    def __init__(
            self,
            locations: npt.NDArray[int],
            adj_list: npt.NDArray[npt.NDArray[int]],
            shape=None,
            dtype=None,
            seed=None
    ):
        super(Network, self).__init__()
        self.locations = locations
        self.adj_list = adj_list

    def sample(self) -> npt.NDArray[int]:
        samples = np.zeros((len(self.locations)))
        for i, loc in enumerate(self.locations):
            print(self.adj_list[loc])
            samples[i] = np.random.choice(self.adj_list[loc], 1)[0]

        return samples

        