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
import sumo_gym.spaces as spaces
from sumo_gym.utils.svg_uitls import vehicle_marker


class FMP(object):
    def __init__(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        n_vertex: int = 0,
        n_depot: int = 1,
        n_charging_station: int = 1,
        n_edge: int = 0,
        n_vehicle: int = 0,
        vertices: sumo_gym.typing.VerticesType = None,
        charging_stations: sumo_gym.typing.FMPChargingStationType = None,
        electric_vehicles: sumo_gym.typing.FMPElectricVehiclesType = None,
        demand: sumo_gym.typing.FMPDemandsType = None,
        edges: sumo_gym.typing.EdgeType = None,
        departures: sumo_gym.typing.DeparturesType = None,
    ):
        """
        :param n_vertex:             the number of vertices
        :param n_depot:              the number of depots
        :param n_charging_station:   the number of charging stations
        :param n_edge:               the number of edges
        :param n_vehicle:            the number of vehicles
        :param vertices:           the vertices, [x_position, y_position]
        :param charging_stations:  the charging stations, [vertex_index, charging_level]
        :param electric_vehicles:  the vehicles, [vehicle_index, charging_level]
        :param demand:             the demand at vertices, [start_vertex_index, end_vertex_index]
        :param edges:              the edges, [from_vertex_index, to_vertex_index]
        :param departures:         the initial settings of vehicles, [starting_vertex_index]
        Create a Fleet Management Problem setting (assume capacity of all vehicles = 1, all demands at each possible node = 1).
        """
        if net_xml_file_path is None or demand_xml_file_path is None:
            # number
            self.n_vertex = n_vertex
            self.n_depot = n_depot
            self.n_edge = n_edge
            self.n_vehicle = n_vehicle
            self.n_charging_station = n_charging_station

            # network
            self.vertices = vertices
            self.charging_stations = charging_stations
            self.depots = np.asarray(range(n_depot)).astype(np.int32)
            self.demand = demand
            self.edges = edges

            # vehicles
            self.electric_vehicles = electric_vehicles
            self.departures = departures

            if not self._is_valid():
                raise ValueError("FMP setting is not valid")

        else:
            self.n_depot = 1 # default value

            # read in the sumo xml files and parse them into FMP initial problem settings
            # self.vertices, self.charging_stations, self.electric_vehicles, self.demand, self.edges, self.departures 
            #      = sumo_gym.utils.decode_xml_fmp(net_xml_file_path, demand_xml_file_path)
            # other settings...

    def __repr__(self):
        return (
            f"Fleet Management problem with {self.n_vertex} vertices, {self.n_depot} depots, {self.n_charging_station} charging station,"
            + f" {self.n_edge} edges, and {self.n_vehicle} vehicles with electric vehicle setting: {self.electric_vehicles}.\n"
            + f"Vertices are {self.vertices};\nDepots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\n"
            + f"Departures are {self.departures}\n"
        )

    def _is_valid(self):
        if (
            not self.n_vertex
            or not self.n_depot
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

    def get_adj_list(self) -> sumo_gym.typing.AdjListType:
        return network_utils.get_adj_list(self.vertices, self.edges)