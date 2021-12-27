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
        mode: str = None,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
        n_vertex: int = 0,
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_electric_vehicle: int = 0,
        n_charging_station: int = 1,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.utils.fmp_utils.Demand = None,
        edges: sumo_gym.typing.EdgeType = None,
        electric_vehicles: sumo_gym.utils.fmp_utils.ElectricVehicles = None,
        departures: sumo_gym.typing.DeparturesType = None,
        charging_stations: sumo_gym.utils.fmp_utils.ChargingStation = None,
    ):
        if mode is None:
            raise Exception("Need a mode to identify")
        elif mode == "sumo_config":
            self.__sumo_config_init(
                net_xml_file_path, demand_xml_file_path, additional_xml_file_path
            )
        elif mode == "numerical":
            self.__numerical_init(
                n_vertex,
                n_edge,
                n_vehicle,
                n_electric_vehicle,
                n_charging_station,
                vertices,
                demand,
                edges,
                electric_vehicles,
                departures,
                charging_stations,
            )
        else:
            raise Exception("Need a valid mode")

        if not self._is_valid():
            raise ValueError("FMP setting is not valid")

    def __numerical_init(
        self,
        n_vertex: int = 0,
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_electric_vehicle: int = 0,
        n_charging_station: int = 1,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.utils.fmp_utils.Demand = None,
        edges: sumo_gym.typing.EdgeType = None,
        electric_vehicles: sumo_gym.utils.fmp_utils.ElectricVehicles = None,
        departures: sumo_gym.typing.DeparturesType = None,
        charging_stations: sumo_gym.utils.fmp_utils.ChargingStation = None,
    ):
        # number
        self.n_vertex = n_vertex
        self.n_edge = n_edge
        self.n_vehicle = n_vehicle
        self.n_electric_vehicle = n_electric_vehicle
        self.n_charging_station = n_charging_station

        # network
        self.vertices = vertices
        self.demand = demand
        self.edges = edges

        # vehicles
        self.electric_vehicles = electric_vehicles
        self.departures = departures
        self.charging_stations = charging_stations

        self.edge_dict = None

    def __sumo_config_init(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
    ):
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
        # `self.vertex_dict` is a mapping from vertex id in SUMO to idx in vertices
        vertices, self.vertex_dict = convert_raw_vertices(raw_vertices)

        # `edges` is a list of Edge instances
        # `self.edge_dict` is a mapping from SUMO edge id to idx in `edges`
        # `self.edge_length_dict` is a dictionary mapping from SUMO edge id to edge length
        (
            edges,
            self.edge_dict,
            self.edge_length_dict,
        ) = convert_raw_edges(raw_edges, self.vertex_dict)

        # `charging_stations` is a list of ChargingStation instances
        # `self.charging_stations_dict` is a mapping from idx in `charging_stations` to SUMO station id
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
        # `self.departures` and `self.actual_departures` are lists of indices in `vertices`
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
        self.n_vehicle = self.n_electric_vehicle = len(self.electric_vehicles)
        self.n_charging_station = len(self.charging_stations)

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


class FMPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    fmp = property(operator.attrgetter("_fmp"))
    __isfrozen = False

    def __init__(self, **kwargs):

        if "mode" not in kwargs:
            raise Exception("Need a mode to identify")
        elif kwargs["mode"] == "sumo_config":
            if "SUMO_GUI_PATH" in os.environ:
                self.sumo_gui_path = os.environ["SUMO_GUI_PATH"]
            else:
                raise Exception("Need 'SUMO_GUI_PATH' in the local environment")

            if "sumo_config_path" in kwargs:
                self.sumo_config_path = kwargs["sumo_config_path"]
                del kwargs["sumo_config_path"]
            else:
                raise Exception("Need 'sumo_config_path' argument to initialize")

            if "render_env" in kwargs:
                self.render_env = kwargs["render_env"]
                del kwargs["render_env"]
            else:
                self.render_env = False

        elif kwargs["mode"] == "numerical":
            if "render_env" in kwargs:
                raise Exception("Only support render for 'sumo_config' mode")

        else:
            raise Exception("Need a valid mode")

        self._fmp = FMP(**kwargs)  # todo: make it "final"
        self.sumo = (
            SumoRender(
                self.sumo_gui_path,
                self.sumo_config_path,
                self.fmp.edge_dict,
                self.fmp.edge_length_dict,
                self.fmp.ev_dict,
                self.fmp.edges,
                self.fmp.n_electric_vehicle,
            )
            if hasattr(self, "render_env") and self.render_env is True
            else None
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
        self.states = [FMPState() for _ in range(self.fmp.n_electric_vehicle)]
        self.responded = set()
        for i in range(self.fmp.n_electric_vehicle):
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
                self.sumo if hasattr(self, "sumo") else None,
            )
        )
        self.actions: sumo_gym.typing.ActionsType = None
        self.rewards: sumo_gym.typing.RewardsType = np.zeros(self.fmp.n_vehicle)

        observation = {
            "Locations": [s.location for s in self.states],
            "Batteries": [s.battery for s in self.states],
            "Is_loading": [s.is_loading for s in self.states],
            "Is_charging": [s.is_charging for s in self.states],
            "Takes_action": [
                True
                if s.is_loading.target == NO_LOADING
                and s.is_charging.target == NO_CHARGING
                else False
                for s in self.states
            ],
        }
        return observation

    def step(self, actions):
        prev_locations = []
        travel_info = []
        self.rewards = np.zeros(self.fmp.n_vehicle)
        for i in range(self.fmp.n_vehicle):
            prev_location = self.states[i].location
            prev_locations.append(prev_location)
            prev_is_loading = self.states[i].is_loading.current
            prev_battery = self.states[i].battery

            self.states[i].battery -= sumo_gym.utils.fmp_utils.dist_between(
                self.fmp.vertices,
                self.fmp.edges,
                prev_location
                if actions[i].location == IDLE_LOCATION
                else actions[i].location,
                prev_location,
            )

            (
                self.states[i].is_loading,
                self.states[i].is_charging,
                self.states[i].location,
            ) = (
                Loading(actions[i].is_loading.current, actions[i].is_loading.target),
                Charging(actions[i].is_charging.current, actions[i].is_charging.target),
                actions[i].location,
            )
            travel_info.append((prev_location, actions[i].location))

            # assert self.states[i].battery >= 0
            if self.states[i].is_charging.current != NO_CHARGING:
                self.states[i].battery += self.fmp.charging_stations[
                    self.states[i].is_charging.current
                ].charging_speed

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
            "Takes_action": [
                True
                if s.is_loading.target == NO_LOADING
                and s.is_charging.target == NO_CHARGING
                and not s.location == IDLE_LOCATION
                else False
                for s in self.states
            ],
        }
        reward, done, info = (
            self.rewards,
            self.responded == set(range(len(self.fmp.demand))),
            "",
        )

        if hasattr(self, "sumo") and self.sumo is not None:
            self.sumo.update_travel_vertex_info_for_vehicle(travel_info)
            self.render()

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
        elif self.sumo is not None:
            self.sumo.render()

    # TODO: need to add default behavior also
    def close(self):
        if hasattr(self, "sumo") and self.sumo is not None:
            self.sumo.close()
