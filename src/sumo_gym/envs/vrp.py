import numpy as np
import numpy.typing as npt
from typing import Type, Tuple, Dict, Any
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sumo_gym
import operator


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
        Create a vehicle routing problem setting.
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

            if not self.is_valid():
                raise ValueError("VRP setting is not valid")

        # else:
        #     self.vertex_num, self.depot_num, self.edge_num, self.vehicle_num, \
        #     self.vertices, self.depots, self.demand, self.edges, self.departures \
        #         = sumo_gym.utils.decoder_xml(net_xml_file_path)

    def __repr__(self):
        return f"Vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures}.\n"

    def is_valid(self):
        if not self.vertex_num or not self.depot_num or not self.edge_num or not self.vehicle_num\
                or self.vertices.any() == None or self.demand.any() == None \
                or self.edges.any() == None or self.departures.any() == None:
            return False
        # todo
        return True


class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    vrp = property(operator.attrgetter("_vrp"))
    action_space = property(operator.attrgetter("_action_space"))
    __isfrozen = False

    def __init__(self, **kwargs):
        self._vrp = VRP(**kwargs)
        self._action_space: npt.NDArray[int] = np.arange(0, self._vrp.vertex_num)

        self.seed()
        self.run = 0
        self.locations: npt.NDArray[Tuple[float]] = self._vrp.departures
        self.action: npt.NDArray[int] = np.asarray([self._action_space[i] for i in
                                                    np.random.randint(self._vrp.depot_num, self._vrp.vertex_num,
                                                                      size=self._vrp.vehicle_num)])
        self.reward = 0.

        self._freeze()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("Cannot add new attributes once instance %r is initialized" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def reset(self):
        self.seed()
        self.run += 1
        self.locations = self._vrp.departures
        self.action = np.asarray([self._action_space[i] for i in
                                  np.random.randint(self._vrp.depot_num, self._vrp.vertex_num,
                                                    size=self._vrp.vehicle_num)])
        self.reward = 0.

    def step(self, action):
        pass

    def render(self, mode='human', close=False):
        # todo: not neccessary
        pass


class CVRP(VRP):
    def __init__(
            self,
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
        Create a vehicle routing problem setting with fixed capacity.
        """
        super(CVRP, self).__init__(vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures)
        self.capacity = capacity


    def __repr__(self):
        return f"Capacitied vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures};\nCapacity are {self.capacity}.\n"

    def is_valid(self):
        if not super(CVRP, self).is_valid():
            return False

        if self.capacity.any() == None:
            return False

        # todo
        return True


class CVRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass