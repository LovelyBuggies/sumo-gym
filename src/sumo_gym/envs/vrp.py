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
            vertex_num: int = 0,
            depot_num: int = 0,
            edge_num: int = 0,
            vehicle_num: int = 0,
            vertices: Dict[int, npt.NDArray[Tuple[np.float64]]] = None,
            demand: Dict[int, np.float64] = None,
            edges: npt.NDArray[Tuple[int]] = None,
            departures: Dict[int, int] = None,
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
        # number
        self.vertex_num: int = vertex_num
        self.depot_num: int = depot_num
        self.edge_num: int = edge_num
        self.vehicle_num: int = vehicle_num

        # network
        self.vertices: Dict[int, npt.NDArray[Tuple[np.float64]]] = vertices
        self.depots: Dict[int, npt.NDArray[Tuple[np.float64]]] = {x: y for x, y in enumerate(list(vertices.values())[0:depot_num])} if vertices is not None else None
        self.demand: Dict[int, np.float64] = demand
        self.edges: npt.NDArray[Tuple[int]] = edges

        # vehicles
        self.departures: Dict[int, int] = departures

    def __repr__(self):
        return f"Vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures}.\n"


class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    vrp = property(operator.attrgetter("_vrp"))
    __isfrozen = False
    def __init__(
            self,
            net_xml_file_path: str = None,
            flow_xml_file_path: str = None,
            **kwargs,
    ):
        if net_xml_file_path is not None and flow_xml_file_path is not None:
            self._vrp = sumo_gym.utils.decoder_xml(net_xml_file_path)
        else:
            self._vrp = VRP(**kwargs)

        self.run = 0
        self.locations: Dict[int, npt.NDArray[Tuple[np.float64]]] = self._vrp.departures
        self.action: Any = None
        self._freeze()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("Cannot add new attributes once instance %r is initialized" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def step(self, action):
        pass

    def reset(self):
        self.run += 1
        self.locations: Dict[int, npt.NDArray[Tuple[np.float64]]] = self._vrp.departures
        self.action: Any = None

    def render(self, mode='human', close=False):
        pass


class CVRP(VRP):
    def __init__(
            self,
            vertex_num: int = 0,
            depot_num: int = 0,
            edge_num: int = 0,
            vehicle_num: int = 0,
            vertices: Dict[int, npt.NDArray[Tuple[np.float64]]] = None,
            demand: Dict[int, np.float64] = None,
            edges: npt.NDArray[Tuple[int]] = None,
            departures: Dict[int, int] = None,
            capacity: Dict[int, npt.NDArray[np.float64]] = None,
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
        self.capacity: Dict[int, npt.NDArray[np.float64]] = capacity

    def __repr__(self):
        return f"Capacitied vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures};\nCapacity are {self.capacity}.\n"


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