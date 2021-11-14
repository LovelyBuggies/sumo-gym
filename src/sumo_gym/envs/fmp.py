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
import sumo_gym.spaces as spaces
from sumo_gym.utils.svg_uitls import vehicle_marker


class FMP(object):
    def __init__(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        n_vertex: int = 0,
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_electric_vehicles: int = 0,
        n_charging_station: int = 1,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.typing.FMPDemandsType = None,
        edges: sumo_gym.typing.EdgeType = None,
        electric_vehicles: sumo_gym.typing.FMPElectricVehiclesType = None,
        departures: sumo_gym.typing.DeparturesType = None,
        charging_stations: sumo_gym.typing.FMPChargingStationType = None,
    ):
        """
        :param n_vertex:                the number of vertices
        :param n_charging_station:      the number of charging stations
        :param n_edge:                  the number of edges
        :param n_vehicle:               the number of vehicles
        :param vertices:                the vertices, [x_position, y_position]
        :param charging_stations:       the charging stations, [vertex_index, charging_level]
        :param electric_vehicles:       the vehicles, [vehicle_index, charging_level]
        :param demand:                  the demand at vertices, [start_vertex_index, end_vertex_index]
        :param edges:                   the edges, [from_vertex_index, to_vertex_index]
        :param departures:              the initial settings of vehicles, [starting_vertex_index]
        Create a Fleet Management Problem setting (assume capacity of all vehicles = 1, all demands at each possible node = 1).
        """
        if net_xml_file_path is None or demand_xml_file_path is None:
            # number
            self.n_vertex = n_vertex
            self.n_edge = n_edge
            self.n_vehicle = n_vehicle
            self.n_electric_vehicles = n_electric_vehicles
            self.n_charging_station = n_charging_station

            # network
            self.vertices = vertices
            self.demand = demand
            self.edges = edges

            # vehicles
            self.electric_vehicles = electric_vehicles
            self.departures = departures
            self.charging_stations = charging_stations

            if not self._is_valid():
                raise ValueError("FMP setting is not valid")

        else:
            pass
            # read in the sumo xml files and parse them into FMP initial problem settings
            # self.vertices, self.charging_stations, self.electric_vehicles, self.demand, self.edges, self.departures 
            #      = sumo_gym.utils.decode_xml_fmp(net_xml_file_path, demand_xml_file_path)
            # other settings...

    def _is_valid(self):
        if (
            not self.n_vertex
            or not self.n_edge
            or not self.n_vehicle
            or not self.n_charging_station
            or self.vertices.any() is None
            or self.charging_stations.any() is None
            or self.electric_vehicles.any() is None
            or self.demand.any() is None
            or self.edges.any() is None
            or self.departures.any() is None
        ):
            return False
        # todo: scale judgement
        return True


class FMPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    vrp = property(operator.attrgetter("_fmp"))
    __isfrozen = False

    def __init__(self, **kwargs):
        self._fmp = FMP(**kwargs)  # todo: make it "final"
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
        self.locations: sumo_gym.typing.LocationsType = self.fmp.departures.astype(int)
        self.batteries: npt.NDArray[float] = np.asarray([ev[-1] for ev in self.fmp.electric_vehicles])
        self.is_loading: npt.NDArray[int] = np.asarray([-1] * self.fmp.electric_vehicles) # -1 means responding no demand, else demand i
        self.is_charging: npt.NDArray[int] = np.asarray([-1] * self.fmp.electric_vehicles) # -1 means not charing ,else charge station i
        self.action_space: spaces.grid.GridSpace = spaces.grid.GridSpace(
            self.fmp.vertices,
            self.fmp.demand,
            self.fmp.edges,
            self.fmp.charging_stations,
            self.locations,
            self.batteries,
            self.is_loading,
            self.is_charging,
        )

        self.actions: sumo_gym.typing.ActionsType = None
        self.rewards: sumo_gym.typing.RewardsType = np.zeros(self.fmp.n_vehicle)

    def step(self, actions):
        for i in range(self.fmp.n_vehicle):
            prev_location = self.locations[i]
            prev_is_loading = self.is_loading[i]
            prev_is_charging = self.is_charging[i]
            self.locations[i] = actions[i][2]
            self.batteries -= self.locations[i] - prev_location
            self.is_loading[i] = actions[i][0]
            self.is_charging[i] = actions[i][1]
            self.rewards[i] -= func(self.locations[i] - prev_location) # should be based on current battery, if low, f is high
            if prev_is_loading != -1 and self.is_loading[i] == -1:
                self.rewards[i] += get_hot_spot_weight(self.fmp.demand[prev_is_loading][0]) * dist_between(self.fmp.demand[prev_is_loading][1], self.fmp.demand[prev_is_loading][0])

            if prev_is_charging != -1 and self.is_charging == -1:
                self.rewards[i] += self.fmp.electric_vehicles[-1]

        observation = {
            "Locations": self.locations,
            "Batteries": self.batteries,
            "Is_loading": self.is_loading,
            "Is_charging": self.is_charging
        }
        reward, done, info = self.rewards, False, ""