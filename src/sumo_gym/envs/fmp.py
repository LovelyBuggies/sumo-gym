import operator
import random
import sys
import os
import numpy as np
import numpy.typing as npt
from typing import Type, Tuple, Dict, Any
import sumo_gym.typing

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sumo_gym
from sumo_gym.utils.sumo_utils import SumoRender
from sumo_gym.utils.svg_uitls import vehicle_marker
from sumo_gym.utils.fmp_utils import *


class FMP(object):
    def __init__(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
        n_vertex: int = 0,
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_electric_vehicles: int = 0,
        n_charging_station: int = 1,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.utils.fmp_utils.Demand = None,
        edges: sumo_gym.typing.EdgeType = None,
        electric_vehicles: sumo_gym.utils.fmp_utils.ElectricVehicles = None,
        departures: sumo_gym.typing.DeparturesType = None,
        charging_stations: sumo_gym.utils.fmp_utils.ChargingStation = None,
    ):
        """
        :param n_vertex:                the number of vertices
        :param n_charging_station:      the number of charging stations
        :param n_edge:                  the number of edges
        :param n_vehicle:               the number of vehicles
        :param vertices:                the vertices, [vertex_index, x_position, y_position]
        :param charging_stations:       the charging stations, [vertex_index, chargeDelay]
        :param electric_vehicles:       the vehicles, [vehicle_index, speed, actualBatteryCapacity, maximumBatteryCapacity)]
        :param demand:                  the demand at vertices, [start_vertex_index, end_vertex_index]
        :param edges:                   the edges, [from_vertex_index, to_vertex_index]
        :param departures:              the initial settings of vehicles, [vehicle_index, starting_vertex_index]
        Create a Fleet Management Problem setting (assume capacity of all vehicles = 1, all demands at each possible node = 1).
        """

        if (
            net_xml_file_path is None
            or demand_xml_file_path is None
            or additional_xml_file_path is None
        ):
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
            (
                raw_vertices,  # [id (str), x_coord (float), y_coord (float)]
                raw_charging_stations,  # [id, (x_coord, y_coord), edge_id, charging speed]
                raw_electric_vehicles,  # [id (str), maximum speed (float), maximumBatteryCapacity (float)]
                raw_edges,  # [id (str), from_vertex_id (str), to_vertex_id (str)]
                raw_departures,  # [vehicle_id, starting_edge_id]
                raw_demand,  # [junction_id, dest_vertex_id]
            ) = sumo_gym.utils.xml_utils.decode_xml_fmp(
                net_xml_file_path, demand_xml_file_path, additional_xml_file_path
            )

            # `vertices` is a list of Vertex instances
            # `self.vertex_dict` is a mapping from
            #   vertex id in SUMO to idx in vertices
            vertices, self.vertex_dict = convert_raw_vertices(raw_vertices)

            # `edges` is a list of Edge instances
            # `self.edge_dict` is a mapping from SUMO edge id
            #   to idx in `edges`
            # `self.edge_length_dict` is a dictionary
            #    mapping from SUMO edge id to edge length
            (
                edges,
                self.edge_dict,
                self.edge_length_dict,
            ) = convert_raw_edges(raw_edges, self.vertex_dict)

            # `charging_stations` is a list of ChargingStation instances
            # `self.charging_stations_dict` is a mapping from idx in `charging_stations`
            #    to SUMO station id
            (
                charging_stations,
                self.charging_stations_dict,
                self.edge_length_dict,
            ) = convert_raw_charging_stations(
                raw_charging_stations,
                vertices,
                edges,
                self.edge_dict,
                self.edge_length_dict,
            )

            # `electric_vehicles` is a list of ElectricVehicles instances
            # `self.ev_dict` is a mapping from ev sumo id to idx in `electric_vehicles`
            electric_vehicles, self.ev_dict = convert_raw_electric_vehicles(
                raw_electric_vehicles
            )

            # departure should be defined for all vehicles
            # `self.departures` and `self.actual_departures` are
            #   lists of indices in `vertices`
            # self.departures[i] is the starting point of electric_vehicles[i] (the endpoint of the passed in edge)
            # self.actual_depatures[i] is the actual start vertex of electric_vehicles[i] (the starting point of the passed in edge)
            departures, actual_departures = convert_raw_departures(
                raw_departures,
                self.ev_dict,
                edges,
                self.edge_dict,
                len(electric_vehicles),
            )

            # `demand` is a list of Demand instances
            demand = convert_raw_demand(raw_demand, self.vertex_dict)

            # set the FMP variables
            self.vertices = np.asarray(vertices)
            self.edges = np.asarray(edges)
            self.charging_stations = np.asarray(charging_stations)
            self.electric_vehicles = np.asarray(electric_vehicles)
            self.departures = np.asarray(departures)
            self.departures = [int(x) for x in self.departures]
            self.actual_departures = np.asarray(actual_departures)
            self.demand = np.asarray(demand)

            self.n_vertex = len(self.vertices)
            self.n_edge = len(self.edges)
            self.n_vehicle = self.n_electric_vehicles = len(self.electric_vehicles)
            self.n_charging_station = len(self.charging_stations)

            if not self._is_valid():
                raise ValueError("FMP setting is not valid")

    def _is_valid(self):
        if (
            not self.n_vertex
            or not self.n_edge
            or not self.n_vehicle
            or not self.n_charging_station
            or self.vertices is None
            or self.charging_stations is None
            or self.electric_vehicles is None
            or self.demand is None
            or self.edges is None
            or self.departures is None
        ):
            return False
        if self.vertices.shape[0] != len(self.vertices):
            return False
        if self.edges.shape[0] != len(self.edges):
            return False
        if self.electric_vehicles.shape[0] != len(self.electric_vehicles):
            return False
        if self.charging_stations.shape[0] != len(self.charging_stations):
            return False
        charging_station_locations = set([cs.location for cs in self.charging_stations])
        for d in self.demand:
            if (
                d.departure in charging_station_locations
                or d.destination in charging_station_locations
            ):
                return False
        # todo: scale judgement
        return True


class FMPState(object):
    def __init__(
        self, location=0, is_loading=Loading(-1, -1), is_charging=-1, battery=0
    ):
        self.location = location
        self.is_loading = is_loading
        self.is_charging = is_charging
        self.battery = battery
        self.stopped = False  # arrive the assigned vertex

    def __repr__(self):
        return (
            f"Location: {self.location}, "
            + f"Is loading: {(self.is_loading.current, self.is_loading.target)},"
            + f"Is charging: {(self.is_charging)} "
            + f"Battery: {(self.battery)}"
        )


class FMPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    fmp = property(operator.attrgetter("_fmp"))
    __isfrozen = False

    def __init__(self, **kwargs):

        if "SUMO_GUI_PATH" in os.environ:
            self.sumo_gui_path = os.environ["SUMO_GUI_PATH"]
        else:
            sys.exit("please declare environment variable 'SUMO_GUI_PATH'")

        if "sumo_configuration_path" in kwargs:
            self.sumo_configuration_path = kwargs["sumo_configuration_path"]
            del kwargs["sumo_configuration_path"]
        else:
            self.sumo_configuration_path = None

        self._fmp = FMP(**kwargs)  # todo: make it "final"

        self.sumo = SumoRender(
            self.sumo_gui_path,
            self.sumo_configuration_path,
            self.fmp.edge_dict,
            self.fmp.edge_length_dict,
            self.fmp.ev_dict,
            self.fmp.edges,
            self.fmp.n_electric_vehicles,
        )

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
        self.states = [FMPState() for _ in range(self.fmp.n_electric_vehicles)]
        self.responded = set()
        for i in range(self.fmp.n_electric_vehicles):
            self.states[i].location = self.fmp.departures[i]
            self.states[i].battery = self.fmp.electric_vehicles[i].capacity

        self.action_space: sumo_gym.spaces.grid.GridSpace = (
            sumo_gym.spaces.grid.GridSpace(
                self.fmp.vertices,
                self.fmp.demand,
                self.responded,
                self.fmp.edges,
                self.fmp.electric_vehicles,
                self.fmp.charging_stations,
                self.states,
                self.sumo,
            )
        )
        self.actions: sumo_gym.typing.ActionsType = None
        self.rewards: sumo_gym.typing.RewardsType = np.zeros(self.fmp.n_vehicle)

        observation = {
            "Locations": [s.location for s in self.states],
            "Batteries": [s.battery for s in self.states],
            "Is_loading": [s.is_loading for s in self.states],
            "Is_charging": [s.is_charging for s in self.states],
        }
        return observation

    def step(self, actions):
        prev_locations = []
        travel_info = []
        for i in range(self.fmp.n_vehicle):
            prev_location = self.states[i].location
            prev_locations.append(prev_location)
            prev_is_loading = self.states[i].is_loading.current
            (
                self.states[i].is_loading,
                self.states[i].is_charging,
                self.states[i].location,
            ) = (
                Loading(actions[i].is_loading.current, actions[i].is_loading.target),
                actions[i].is_charging.charging_station,
                actions[i].location,
            )
            self.states[i].battery -= sumo_gym.utils.fmp_utils.dist_between(
                self.fmp.vertices,
                self.fmp.edges,
                self.states[i].location,
                prev_location,
            )
            travel_info.append((prev_location, actions[i].location))

            assert self.states[i].battery >= 0
            self.states[i].battery += actions[i].is_charging.battery_charged
            self.rewards[i] += actions[i].is_charging.battery_charged

            if prev_is_loading != -1 and self.states[i].is_loading.current == -1:
                self.responded.add(prev_is_loading)

                self.rewards[i] += sumo_gym.utils.fmp_utils.get_hot_spot_weight(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.demand,
                    self.fmp.demand[prev_is_loading].departure,
                ) * sumo_gym.utils.fmp_utils.dist_between(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.demand[prev_is_loading].departure,
                    self.fmp.demand[prev_is_loading].destination,
                )

        print("Batteries:", [s.battery for s in self.states])
        print("Rewards:", self.rewards)
        observation = {
            "Locations": [s.location for s in self.states],
            "Batteries": [s.battery for s in self.states],
            "Is_loading": [s.is_loading for s in self.states],
            "Is_charging": [s.is_charging for s in self.states],
        }
        reward, done, info = (
            self.rewards,
            self.responded == set(range(len(self.fmp.demand))),
            "",
        )

        self.sumo.update_travel_vertex_info_for_vehicle(travel_info)

        return observation, reward, done, info

    def plot(
        self,
        *,
        ax_dict=None,
        **kwargs: Any,
    ) -> Any:
        get_colors = lambda n: list(
            map(lambda i: "#" + "%06x" % random.randint(0x000000, 0x666666), range(n))
        )
        plot_kwargs = {
            "fmp_vertex_s": 200,
            "fmp_vertex_c": "navy",
            "fmp_vertex_marker": r"$\odot$",
            "demand_width": 0.4,
            "demand_color": get_colors(self.fmp.n_vertex),
            "loading_width": 0.6,
            "loading_color": get_colors(self.fmp.n_vehicle),
            "location_marker": vehicle_marker,
            "location_s": 2000,
            "location_c": "lightgrey",
        }
        self.plot(**plot_kwargs)

    def render(self, mode="human"):
        # todo: this part should be move to .render()
        if self.sumo_gui_path is None:
            raise EnvironmentError("Need sumo-gui path to render")
        else:
            self.sumo.render()

    # TODO: need to add default behavior also
    def close(self):
        self.sumo.close()
        pass
