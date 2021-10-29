import numpy as np
import numpy.typing as npt
from typing import Type, Tuple, Dict, Any
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sumo_gym.utils.xml_converter import encode_xml, decoder_xml


class VRP(object):
    def __init__(self, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures):
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
        self.depots: Dict[int, npt.NDArray[Tuple[np.float64]]] = {x: y for x, y in enumerate(list(vertices.values())[0:depot_num])}
        self.demand: Dict[int, np.float64] = demand
        self.edges: npt.NDArray[Tuple[int]] = edges

        # vehicles
        self.departures: Dict[int, int] = departures

    def __repr__(self):
        return f"Vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures}.\n"


class VRPState(VRP):
    def __init__(self, id, parent, locations, action):
        """
        :param id:          The state id
        :param parent:      The parent
        :param locations:   The current locations of vehicles
        :param action:      The action took in this state
        A state keeps dynamic features of VRP during evolving.
        """
        self.id: int = id
        self.parent: Type[VRPState] = parent
        self.locations: Dict[int, npt.NDArray[Tuple[np.float64]]] = locations
        self.action: Any = action


class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, net_xml_path=None, flow_xml_path=None):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass


class CVRP(VRP):
    def __init__(self, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures, capacity):
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


class CVRPState(CVRP):
    def __init__(self, id, parent, locations, action, load):
        """
        :param id:          The state id
        :param parent:      The parent
        :param locations:   The current locations of vehicles
        :param action:      The action took in this state
        :param load:        The load of vehicles currently
        A state keeps dynamic features of VRP during evolving.
        """
        self.id: int = id
        self.parent: Type[VRPState] = parent
        self.locations: Dict[int, npt.NDArray[Tuple[np.float64]]] = locations
        self.action: Any = action
        self.load: Dict[int, npt.NDArray[np.float64]] = load


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