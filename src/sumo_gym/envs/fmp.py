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
from sumo_gym.utils.svg_uitls import vehicle_marker
from sumo_gym.utils.fmp_utils import (
    Vertex,
    Edge,
    Demand,
    Loading,
    ChargingStation,
    ElectricVehicles,
)


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
                raw_charging_stations, # [id, (x_coord, y_coord), edge_id, charging speed]
                raw_electric_vehicles, # [id (str), maximum speed (float), maximumBatteryCapacity (float)]
                raw_edges,  # [id (str), from_vertex_id (str), to_vertex_id (str)]
                raw_departures,  # [vehicle_id, starting_edge_id]
                raw_demand,  # [junction_id, dest_vertex_id]
            ) = sumo_gym.utils.xml_utils.decode_xml_fmp(
                net_xml_file_path, demand_xml_file_path,
                additional_xml_file_path
            )

            self.vertices = []
            self.vertex_dict = {} # vertex id to idx in self.vertices
            counter = 0
            for v in raw_vertices:
                self.vertices.append(Vertex(v[1], v[2]))
                self.vertex_dict[v[0]] = counter
                counter += 1

            self.edges = []
            self.edge_dict = {} # sumo edge_id to idx in self.edges
            self.revert_edge_dict = {} # Edge instance to sumo edge_id
            counter = 0
            for e in raw_edges:
                new_edge = Edge(self.vertex_dict[e[1]], self.vertex_dict[e[2]])
                self.edges.append(new_edge)
                self.revert_edge_dict[new_edge] = e[0]
                self.edge_dict[e[0]] = counter
                counter += 1

            self.charging_dict = {} # CS id to idx in self.charging_stations
            self.station_to_sumo_edge_id = {} # charging_station idx to original sumo edge_id
            edge_idx_delete = []
            self.charging_stations = []
            counter = 0
            vtx_counter = len(self.vertices)
            for charging_station in raw_charging_stations:
                
                self.charging_dict[charging_station[0]] = counter
                self.station_to_sumo_edge_id[counter] = charging_station[0]
                
                # create new vertex with charging station's location
                x_coord, y_coord = charging_station[1]
                new_vtx = Vertex(x_coord, y_coord)
                self.vertices.append(new_vtx)
                
                # keep a list of edge indices to delete later
                edge_id = charging_station[2]
                edge_idx_delete.append(self.edge_dict[edge_id])

                # create two new edges
                # first get the start and end vertices of the old edge
                old_edge_start_idx = self.edges[self.edge_dict[edge_id]].start
                old_edge_end_idx = self.edges[self.edge_dict[edge_id]].end

                new_edge1 = Edge(old_edge_start_idx, vtx_counter)
                new_edge2 = Edge(vtx_counter, old_edge_end_idx)
                self.edges.append(new_edge1)
                self.edges.append(new_edge2)

                # instantiate new ChargingStation with location set to vtx_counter
                self.charging_stations.append(ChargingStation(vtx_counter, 220, charging_station[3]))
                
                vtx_counter += 1
                counter += 1

            self.charging_stations = np.array(self.charging_stations)
            self.vertices = np.array(self.vertices)

            # remove indices from self.edges and fix self.edge_dict
            # sort the indices in reverse order and pop the indices
            # since the only elements in the list with different indices
            # after a pop are those that reside on indices after the popped index
            edge_idx_delete = sorted(list(set(edge_idx_delete)), reverse=True)
            for idx in edge_idx_delete:
                self.edges.pop(idx)
            self.edges = np.array(self.edges)

            # now, we need to repopulate self.edge_dict since the indices have changed
            self.edge_dict = {} # sumo edge_id to idx in self.edges
            counter = 0
            for edge in self.edges:
                if edge in self.revert_edge_dict:
                    sumo_edge_id = self.revert_edge_dict[edge]
                    self.edge_dict[sumo_edge_id] = counter
                counter += 1

            self.electric_vehicles = []
            self.ev_dict = {} # ev id to idx in self.electric_vehicles
            counter = 0
            for vehicle in raw_electric_vehicles:
                self.electric_vehicles.append(ElectricVehicles(counter, vehicle[1], 220, vehicle[2]))
                self.ev_dict[vehicle[0]] = counter
                counter += 1
            self.electric_vehicles = np.array(self.electric_vehicles)

            # departure should be defined for all vehicles, so
            # self.departures[i] should be the edge idx in self.edges of self.electric_vehicles[i]
            self.departures = np.zeros(len(self.electric_vehicles))
            for dpt in raw_departures:
                self.departures[self.ev_dict[dpt[0]]] = self.edges[self.edge_dict[dpt[1]]].start
            self.departures = np.array(self.departures)
            self.departures = [int(num) for num in self.departures]
            
            self.demand = []
            for d in raw_demand:
                self.demand.append(Demand(self.vertex_dict[d[0]], self.vertex_dict[d[1]]))
            self.demand = np.array(self.demand)


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
            )
        )
        self.actions: sumo_gym.typing.ActionsType = None
        self.rewards: sumo_gym.typing.RewardsType = np.zeros(self.fmp.n_vehicle)

    def step(self, actions):
        for i in range(self.fmp.n_vehicle):
            prev_location = self.states[i].location
            prev_is_loading = self.states[i].is_loading.current
            prev_battery = self.states[i].battery
            (
                self.states[i].is_loading,
                self.states[i].is_charging,
                self.states[i].location,
            ) = (
                Loading(actions[i].is_loading.current, actions[i].is_loading.target),
                actions[i].is_charging,
                actions[i].location,
            )
            self.states[i].battery -= sumo_gym.utils.fmp_utils.dist_between(
                self.fmp.vertices,
                self.fmp.edges,
                self.states[i].location,
                prev_location,
            )
            assert self.states[i].battery >= 0
            if self.states[i].is_charging != -1:
                self.states[i].battery += self.fmp.charging_stations[
                    self.states[i].is_charging
                ].charging_speed

            self.rewards[i] += min(self.states[i].battery - prev_battery, 0)

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
        return observation, reward, done, info

    def plot(
        self,
        *,
        ax_dict=None,
        **kwargs: Any,
    ) -> Any:
        import sumo_gym.plot

        return sumo_gym.plot.plot_FMPEnv(self, ax_dict=ax_dict, **kwargs)

    def render(self, mode="human"):
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
