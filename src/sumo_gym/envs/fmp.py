import operator
import os
import sumo_gym.typing

import gym
import sumo_gym
from sumo_gym.utils.sumo_utils import SumoRender
from sumo_gym.utils.fmp_utils import *

import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


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
        self.ev_dict = {f"v{i}": i for i, _ in enumerate(self.electric_vehicles)}
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


class FMPEnv(AECEnv):
    metadata = {"render.modes": ["human"]}
    fmp = property(operator.attrgetter("_fmp"))

    def __init__(self, **kwargs):
        """
        Initialize FMPEnv.
        1. Setup render variables
        2. Setup FMP variables
        3. Setup Petting-Zoo environment variables (same as reset)
        """

        # setup render variables
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

        # setup FMP variables
        self._fmp = FMP(**kwargs)
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

        # setup Petting-Zoo environment variables
        self.possible_agents = list(self.fmp.ev_dict.keys())
        self.agent_name_mapping = self.fmp.ev_dict

        self._action_spaces = {
            agent: gym.spaces.Discrete(
                self.fmp.n_charging_station + len(self.fmp.demand) + 1
            )
            for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                high=np.array(
                    [
                        self.fmp.n_vertex,
                        self.fmp.electric_vehicles[0].capacity,
                        2 * len(self.fmp.demand) + 1,
                        2 * self.fmp.n_charging_station + 1,
                        self.fmp.n_charging_station + len(self.fmp.demand) + 1,
                    ]
                ),
                dtype=np.float64,
            )
            for agent in self.possible_agents
        }

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.dones = {agent: False for agent in self.agents}
        self.responded = {agent: list() for agent in self.agents}

        self.infos = {agent: dict() for agent in self.agents}

        self.states = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array(
                [
                    self.fmp.n_vertex,
                    self.fmp.electric_vehicles[0].capacity,
                    2 * len(self.fmp.demand) + 1,
                    2 * self.fmp.n_charging_station + 1,
                    self.fmp.n_charging_station + len(self.fmp.demand) + 1,
                ]
            ),
            dtype=np.float64,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(
            self.fmp.n_charging_station + len(self.fmp.demand) + 1
        )

    def observe(self, agent):
        return np.array(self.observations[agent])

    def reset(self):
        """
        Reset Petting-Zoo environment variables.
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.dones = {agent: False for agent in self.agents}
        self.responded = {agent: list() for agent in self.agents}

        self.infos = {agent: dict() for agent in self.agents}

        self.states = {
            agent: [
                self.fmp.departures[self.agent_name_mapping[agent]],
                self.fmp.electric_vehicles[self.agent_name_mapping[agent]].capacity,
                0,
                0,
            ]
            for agent in self.agents
        }
        self.observations = {
            agent: self.states[agent][:4] + [None] for agent in self.agents
        }

        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observations

    def step(self, action):
        """
        Step takes an action for the current agent.
        Update of state, normal reward, responded, observation are in state "move" and "transition".
        Update of done, broken reward, broken responded and agent are in step.
        """
        agent = self.agent_selection

        if self.dones[agent]:
            # if done, remove it
            self._was_done_step(None)
            self.observations[agent][4] = None
            return self.observations, self.rewards, self.dones, self.infos

        else:
            self.rewards[agent] = 0
            # state move
            if self.states[agent][2] != 0 or self.states[agent][3] != 0:
                self._state_move(agent)
            # state transition
            else:
                self._state_transition(action)

            # judge whether it's done
            self.dones[agent] = set(
                [d for r in self.responded.values() for d in r]
            ) == set(range(len(self.fmp.demand)))

            # check whether is in negative battery
            if self.states[agent][1] < 0:
                self.dones[agent] = True
                if self.states[agent][2] != 0:
                    dmd_idx = (
                        self.states[agent][2] - len(self.fmp.demand) - 1
                        if self.states[agent][2] > len(self.fmp.demand)
                        else self.states[agent][2] - 1
                    )
                    print(self.responded[agent], dmd_idx)
                    del self.responded[agent][
                        len(self.responded[agent])
                        - self.responded[agent][::-1].index(dmd_idx)
                        - 1
                    ]

                self.rewards[agent] -= 1000

            self.cumulative_rewards[agent] += self.rewards[agent]
            print(self.observations[agent], self.rewards[agent], self.cumulative_rewards[agent], self.responded[agent])
            if self._agent_selector.is_last():
                self.num_moves += 1
                print("------------------------------")

            self.agent_selection = self._agent_selector.next()

            return self.observations, self.rewards, self.dones, self.infos

    def _state_move(self, agent):
        """
        Deal with moving state:
        1. Move the state
        2. Update observation (force to change observations' action) and reward, accordingly
        """
        # if responding
        if self.states[agent][2] != 0:
            if self.states[agent][2] > len(self.fmp.demand):
                dmd_idx = self.states[agent][2] - len(self.fmp.demand) - 1
                dest_loc = self.fmp.demand[dmd_idx].destination
                print("Move: ", agent, " is in responding demand ", dmd_idx)

                self.states[agent][0] = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.states[agent][0],
                    dest_loc,
                )
                self.states[agent][1] -= 1
                if self.states[agent][0] == dest_loc:
                    self.states[agent][2] = 0
                    self.rewards[agent] += (
                        sumo_gym.utils.fmp_utils.get_hot_spot_weight(
                            self.fmp.vertices,
                            self.fmp.edges,
                            self.fmp.demand,
                            self.fmp.demand[dmd_idx].departure,
                        )
                        * sumo_gym.utils.fmp_utils.dist_between(
                            self.fmp.vertices,
                            self.fmp.edges,
                            self.fmp.demand[dmd_idx].departure,
                            self.fmp.demand[dmd_idx].destination,
                        )
                        if self.responded[agent].count(dmd_idx) == 1
                        else 0
                    )

                self.rewards[agent] -= 1
            else:
                dmd_idx = self.states[agent][2] - 1
                dest_loc = self.fmp.demand[dmd_idx].departure
                print("Move: ", agent, " is to respond demand ", dmd_idx)

                self.states[agent][0] = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.states[agent][0],
                    dest_loc,
                )
                self.states[agent][1] -= 1
                if self.states[agent][0] == dest_loc:
                    self.states[agent][2] += len(self.fmp.demand)

                self.rewards[agent] -= 1

        # if charging
        elif self.states[agent][3] != 0:
            if self.states[agent][3] > self.fmp.n_charging_station:
                cs_idx = self.states[agent][3] - self.fmp.n_charging_station - 1
                print("Move: ", agent, " is in charging at ", cs_idx)
                self.states[agent][1] += self.fmp.charging_stations[
                    cs_idx
                ].charging_speed
                if (
                    self.states[agent][1]
                    >= self.fmp.electric_vehicles[
                        self.agent_name_mapping[agent]
                    ].capacity
                ):
                    self.states[agent][3] = 0
            else:
                cs_idx = self.states[agent][3] - 1
                dest_loc = self.fmp.charging_stations[cs_idx].location
                print("Move: ", agent, "is to go to charge at ", cs_idx)

                self.states[agent][0] = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.states[agent][0],
                    dest_loc,
                )
                self.states[agent][1] -= 1

                if self.states[agent][0] == dest_loc:
                    self.states[agent][3] += self.fmp.n_charging_station

                self.rewards[agent] -= 1

        # not charging and not loading should not be moving
        else:
            raise ValueError("Agent that not responding or charging should not move")

        self.observations[agent][:4] = self.states[agent][:4]
        self.observations[agent][4] = 0

    def _state_transition(self, action):
        """
        Transit the state of current agent according to the action and update observation:
        1. Update states (if "move" action for a state need to make decision, then no change)
        2. Update responded if responding a new demand
        3. Update observation and reward, accordingly
        """
        agent = self.agent_selection

        if action == 0:
            print("Trans: ", agent, "is taking moving action")
            pass

        # action to charge
        elif action <= self.fmp.n_charging_station:
            print("Trans: ", agent, "is to go to charge at ", action - 1)
            self.states[agent] = [
                one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.states[agent][0],
                    self.fmp.charging_stations[action - 1].location,
                ),
                self.states[agent][1] - 1,
                0,
                action,
            ]
            self.rewards[agent] = -1

        # action to load
        else:
            print(
                "Trans: ",
                agent,
                " is to respond demand ",
                action - self.fmp.n_charging_station - 1,
            )
            self.states[agent] = [
                one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.states[agent][0],
                    self.fmp.demand[action - self.fmp.n_charging_station - 1].departure,
                ),
                self.states[agent][1] - 1,
                action - self.fmp.n_charging_station,
                0,
            ]
            self.rewards[agent] = -1
            self.responded[agent].append(self.states[agent][2] - 1)

        self.observations[agent][:4] = self.states[agent][:4]

    def render(self, mode="human"):
        if self.sumo_gui_path is None:
            raise EnvironmentError("Need sumo-gui path to render")
        elif self.sumo is not None:
            # TODO: need sumo render here
            print("sumo render")

    def close(self):
        if hasattr(self, "sumo") and self.sumo is not None:
            self.sumo.close()
