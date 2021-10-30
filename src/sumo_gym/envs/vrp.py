import operator
import numpy as np
import numpy.typing as npt
from typing import Type, Tuple, Any
import sumo_gym.typing

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
            vertices: sumo_gym.typing.VerticesType = None,
            demand: sumo_gym.typing.DemandType = None,
            edges: sumo_gym.typing.EdgeType = None,
            departures: sumo_gym.typing.DeparturesType = None,
            capacity: sumo_gym.typing.CapacityType = None,
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
            self.depots = np.asarray(range(depot_num)).astype(np.int32)
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

    def get_adj_list(self) -> sumo_gym.typing.AdjListType:
        return network_utils.get_adj_list(self.vertices, self.edges)

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    vrp = property(operator.attrgetter("_vrp"))
    __isfrozen = False

    def __init__(self, **kwargs):
        self._vrp = VRP(**kwargs) # todo: make it "final"
        self.run = -1
        self._reset()
        self._freeze()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("Cannot add new attributes once instance %r is initialized" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def reset(self):
        return self._reset()

    def _reset(self):
        self.run += 1
        self.locations: sumo_gym.typing.LocationsType = self.vrp.departures.astype(int)
        self.loading: sumo_gym.typing.LoadingType = np.zeros(self.vrp.vehicle_num)
        self.action_space: spaces.network.NetworkSpace = spaces.network.NetworkSpace(
            self.locations,
            self.vrp.get_adj_list(),
            self.vrp.demand,
            np.asarray([False] * self.vrp.vehicle_num),
            self.vrp.depots,
        )
        self.actions: sumo_gym.typing.ActionsType = None
        self.rewards: sumo_gym.typing.RewardsType = np.zeros(self.vrp.vehicle_num)

        return {"Loading": self.loading, "Locations": self.locations,}

    def step(self, actions):
        vehicle_num = self.vrp.vehicle_num
        prev_location = self.locations
        self.locations = actions
        fully_loaded = np.asarray([False] * vehicle_num)
        for i in range(vehicle_num):
            capacity_remaining = self.vrp.capacity[i] - self.loading[i]
            location = int(self.locations[i])
            load_amount = min(self.vrp.demand[location], capacity_remaining)
            fully_loaded[i] = False if capacity_remaining > 0 else True
            self.loading[i] += load_amount
            self.vrp.demand[location] -= load_amount
            self.rewards[i] += 20 * load_amount # todo: make rewards_rate more customized
            self.rewards[i] -= sumo_gym.utils.calculate_dist(prev_location[i], location, self.vrp.vertices)

        self.action_space: spaces.network.NetworkSpace = spaces.network.NetworkSpace(
            self.locations,
            self.vrp.get_adj_list(),
            self.vrp.demand,
            fully_loaded,
            self.vrp.depots,
        )

        observation = {
            "Loading": self.loading,
            "Locations": self.locations,
        }
        reward, done, info = self.rewards, True, ""
        for l in self.locations:
            if l not in self.vrp.depots:
                done = False
                break

        done = all(self.vrp.demand) == 0 and done
        return observation, reward, done, info

