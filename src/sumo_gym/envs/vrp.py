import operator
import numpy as np
import random
import numpy.typing as npt
from typing import Type, Tuple, Dict, Any
import sumo_gym.typing

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sumo_gym
import sumo_gym.utils.network_utils as network_utils
from sumo_gym.utils.svg_uitls import vehicle_marker


class VRP(object):
    def __init__(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        n_vertex: int = 0,
        n_depot: int = 1,
        n_edge: int = 0,
        n_vehicle: int = 0,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.typing.DemandType = None,
        edges: sumo_gym.typing.EdgeType = None,
        departures: sumo_gym.typing.DeparturesType = None,
        capacity: sumo_gym.typing.CapacityType = None,
    ):
        """
        :param n_vertex:      the number of vertices
        :param n_depot:       the number of depots
        :param n_edge:        the number of edges
        :param n_vehicle:     the number of vehicles
        :param vertices:        the vertices
        :param demand:          the demand of vertices
        :param edges:           the edges
        :param departures:      the departures of vehicles
        :param capacity:        the capacity of vehicles
        Create a vehicle routing problem setting (CVRP if capacity is activated).
        """
        if net_xml_file_path is None or demand_xml_file_path is None:
            # number
            self.n_vertex = n_vertex
            self.n_depot = n_depot
            self.n_edge = n_edge
            self.n_vehicle = n_vehicle

            # network
            self.vertices = vertices
            self.depots = np.asarray(range(n_depot)).astype(np.int32)
            self.demand = demand
            self.edges = edges

            # vehicles
            self.departures = departures
            self.capacity = (
                np.asarray([float("inf") for _ in range(n_vehicle)])
                if capacity is None
                else capacity
            )

            if not self._is_valid():
                raise ValueError("VRP setting is not valid")

        else:
            self.n_depot = 1  # default value

            # read in the sumo xml files and parse them into VRP initial problem settings
            (
                self.vertices,
                self.demand,
                self.edges,
                self.departures,
                self.capacity,
            ) = sumo_gym.utils.decode_xml(net_xml_file_path, demand_xml_file_path)
            self.n_vertex, _ = self.vertices.shape
            self.n_edge, _ = self.edges.shape
            self.n_vehicle = self.departures.shape[0]
            self.depots = self.vertices[: self.n_depot]

    def __repr__(self):
        return (
            f"Vehicle routing problem with {self.n_vertex} vertices, {self.n_depot} depots,"
            + f" {self.n_edge} edges, and {self.n_vehicle} vehicles.\nVertices are {self.vertices};\n"
            + f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are"
            + f" {self.departures};\nCapacity are {self.capacity}.\n"
        )

    def _is_valid(self):
        if (
            not self.n_vertex
            or not self.n_depot
            or not self.n_edge
            or not self.n_vehicle
            or self.vertices.any() is None
            or self.demand.any() is None
            or self.edges.any() is None
            or self.departures.any() is None
            or self.capacity.any() is None
        ):
            return False
        # todo: scale judgement
        return True

    def get_adj_list(self) -> sumo_gym.typing.AdjListType:
        return network_utils.get_adj_list(self.vertices, self.edges)


class VRPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    vrp = property(operator.attrgetter("_vrp"))
    __isfrozen = False

    def __init__(self, **kwargs):
        self._vrp = VRP(**kwargs)  # todo: make it "final"
        self.run = -1
        self._reset()
        self._freeze()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(
                "Cannot add new attributes once instance %r is initialized" % self
            )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def reset(self):
        return self._reset()

    def _reset(self):
        self.run += 1
        self.locations: sumo_gym.typing.LocationsType = self.vrp.departures.astype(int)
        self.loading: sumo_gym.typing.LoadingType = np.zeros(self.vrp.n_vehicle)
        self.action_space: sumo_gym.spaces.network.NetworkSpace = (
            sumo_gym.spaces.network.NetworkSpace(
                self.locations,
                self.vrp.get_adj_list(),
                self.vrp.demand,
                np.asarray([False] * self.vrp.n_vehicle),
                self.vrp.depots,
            )
        )
        self.actions: sumo_gym.typing.ActionsType = None
        self.rewards: sumo_gym.typing.RewardsType = np.zeros(self.vrp.n_vehicle)

        return {
            "Loading": self.loading,
            "Locations": self.locations,
        }

    def step(self, actions):
        n_vehicle = self.vrp.n_vehicle
        prev_location = self.locations
        self.locations = actions
        fully_loaded = np.asarray([False] * n_vehicle)
        for i in range(n_vehicle):
            capacity_remaining = self.vrp.capacity[i] - self.loading[i]
            location = int(self.locations[i])
            if location in self.vrp.depots:
                fully_loaded[i] = False
                self.loading[i] = 0
            else:
                load_amount = min(self.vrp.demand[location], capacity_remaining)
                fully_loaded[i] = False if capacity_remaining > 0 else True
                self.loading[i] += load_amount
                self.vrp.demand[location] -= load_amount
                self.rewards[i] += (
                    5 * load_amount
                )  # todo: make rewards_rate more customized
                self.rewards[i] -= sumo_gym.utils.calculate_dist(
                    prev_location[i], location, self.vrp.vertices
                )

        self.action_space: sumo_gym.spaces.network.NetworkSpace = (
            sumo_gym.spaces.network.NetworkSpace(
                self.locations,
                self.vrp.get_adj_list(),
                self.vrp.demand,
                fully_loaded,
                self.vrp.depots,
            )
        )

        observation = {
            "Loading": self.loading,
            "Locations": self.locations,
        }
        reward, done = self.rewards, True
        for l in self.locations:
            if l not in self.vrp.depots:
                done = False
                break

        done &= all([True if d == 0 else False for d in self.vrp.demand])
        info = (
            f"Action: {actions}; \nDemand: {self.vrp.demand.astype(int)}"
            + f"; \nReward: {self.rewards.astype(int)}.\n"
        )
        return observation, reward, done, info

    def plot(
        self,
        *,
        ax_dict=None,
        **kwargs: Any,
    ) -> Any:
        import sumo_gym.plot

        return sumo_gym.plot.plot_VRPEnv(self, ax_dict=ax_dict, **kwargs)

    def render(self, mode="human"):
        get_colors = lambda n: list(
            map(lambda i: "#" + "%06x" % random.randint(0x000000, 0x666666), range(n))
        )
        plot_kwargs = {
            "vrp_depot_s": 200,
            "vrp_vertex_s": 200,
            "vrp_depot_c": "darkgreen",
            "vrp_vertex_c": "navy",
            "vrp_depot_marker": r"$\otimes$",
            "vrp_vertex_marker": r"$\odot$",
            "demand_width": 0.4,
            "demand_color": get_colors(self.vrp.n_vertex),
            "loading_width": 0.6,
            "loading_color": get_colors(self.vrp.n_vehicle),
            "location_marker": vehicle_marker,
            "location_s": 2000,
            "location_c": "lightgrey",
        }
        self.plot(**plot_kwargs)
