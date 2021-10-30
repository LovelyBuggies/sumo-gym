import operator
import numpy as np
import numpy.typing as npt
from typing import Type, Tuple, Any
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sumo_gym
import sumo_gym.utils.network_utils as network_utils
import sumo_gym.spaces as spaces


class VRP(object):
    def __init__(
            self,
            net_xml_file_path: str = None,
            flow_xml_file_path: str = None,
            vertex_num: int = 0,
            depot_num: int = 0,
            edge_num: int = 0,
            vehicle_num: int = 0,
            vertices: npt.NDArray[Tuple[float]] = None,
            demand: npt.NDArray[float] = None,
            edges: npt.NDArray[Tuple[int]] = None,
            departures: npt.NDArray[int] = None,
            capacity: npt.NDArray[float] = None,
    ):
        """
        :param vertex_num:      the number of vertices
        :param depot_num:       the number of depots
        :param edge_num:        the number of edges
        :param vehicle_num:     the number of vehicles
        :param vertices:        the vertices
        :param demand:          the demand of vertices
        :param edges:           the edges
        :param departures:      the departures of vehicles
        :param capacity:        the capacity of vehicles
        Create a vehicle routing problem setting (CVRP if capacity is activated).
        """
        if net_xml_file_path is None or flow_xml_file_path is None:
            # number
            self.vertex_num = vertex_num
            self.depot_num = depot_num
            self.edge_num = edge_num
            self.vehicle_num = vehicle_num

            # network
            self.vertices = vertices
            self.depots = vertices[0:depot_num] if vertices is not None else None
            self.demand = demand
            self.edges = edges

            # vehicles
            self.departures = departures
            self.capacity = np.asarray([float('inf') for _ in range(vehicle_num)]) if capacity is None else capacity

            if not self._is_valid():
                raise ValueError("VRP setting is not valid")

        # else:
        #     self.vertex_num, self.depot_num, self.edge_num, self.vehicle_num, \
        #     self.vertices, self.depots, self.demand, self.edges, self.departures \
        #         = sumo_gym.utils.decoder_xml(net_xml_file_path)

    def __repr__(self):
        return f"Vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures};\nCapacity are {self.capacity}.\n"

    def _is_valid(self):
        if not self.vertex_num or not self.depot_num or not self.edge_num or not self.vehicle_num\
                or self.vertices.any() is None or self.demand.any() is None \
                or self.edges.any() is None or self.departures.any() is None\
                or self.capacity.any() is None:
            return False
        # todo: scale judgement
        return True

    def get_adj_list(self) -> npt.NDArray[npt.NDArray[int]]:
        return network_utils.get_adj_list(self.vertices, self.edges)

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    vrp = property(operator.attrgetter("_vrp"))
    __isfrozen = False

    def __init__(self, **kwargs):
        self._vrp = VRP(**kwargs)
        self.run = 0

        self.locations: npt.NDArray[int] = self.vrp.departures
        self.loading: npt.NDArray[Tuple[float]] = np.zeros(self.vrp.vehicle_num)
        self.action_space: spaces.network.Network = spaces.network.Network(self.locations, self.vrp.get_adj_list())
        self.actions: npt.NDArray[int] = self.action_space.sample()
        self.rewards = np.zeros(self.vrp.vehicle_num)
        self._freeze()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("Cannot add new attributes once instance %r is initialized" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def reset(self):
        self.run += 1
        self.locations: npt.NDArray[int] = self.vrp.departures
        self.loading: npt.NDArray[Tuple[float]] = np.zeros(self.vrp.vehicle_num)
        self.action_space: spaces.network.Network = spaces.network.Network(self.locations, self.vrp.get_adj_list())
        self.actions: npt.NDArray[int] = self.action_space.sample()
        self.rewards = np.zeros(self.vrp.vehicle_num)

    def step(self, actions):
        # prev_location = self.locations
        # prev_loading = self.loading
        # vehicle_num = self.vrp.vehicle_num
        # self.locations = actions
        # # Todo: what if fully loaded
        # self.loading = np.asarray([
        #     self.loading[i] + \
        #     min(self.vrp.demand[self.locations[i]], self.vrp.capacity[i] - self.loading[i]) \
        #     for i in range(vehicle_num)
        # ])
        # # todo: make rewards_rate more customized
        # self.rewards += [20 * max(0, self.loading[i] - prev_loading[i]) \
        #                     - sumo_gym.utils.calculate_dist(prev_location[i], locations[i])
        #                     for i in range(vehicle_num)]
        pass