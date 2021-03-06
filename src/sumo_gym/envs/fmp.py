from itertools import chain
import operator
import os
import random

from black import prev_siblings_are
import sumo_gym.typing

import gym
import sumo_gym
from sumo_gym.utils.sumo_utils import SumoRender
from sumo_gym.utils.fmp_utils import *

import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from statistics import mean


class FMP(object):
    def __init__(
        self,
        mode: str = None,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
        n_vertex: int = 0,
        n_area: int = 0,
        n_demand: int = 0,
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_electric_vehicle: int = 0,
        n_charging_station: int = 1,
        vertices: sumo_gym.typing.VerticesType = None,
        demands: sumo_gym.utils.fmp_utils.Demand = None,
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
                n_area,
                n_demand,
                n_edge,
                n_vehicle,
                n_electric_vehicle,
                n_charging_station,
                vertices,
                demands,
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
        n_area: int = 0,
        n_demand: int = 0,
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_electric_vehicle: int = 0,
        n_charging_station: int = 1,
        vertices: sumo_gym.typing.VerticesType = None,
        demands: sumo_gym.utils.fmp_utils.Demand = None,
        edges: sumo_gym.typing.EdgeType = None,
        electric_vehicles: sumo_gym.utils.fmp_utils.ElectricVehicles = None,
        departures: sumo_gym.typing.DeparturesType = None,
        charging_stations: sumo_gym.utils.fmp_utils.ChargingStation = None,
    ):
        # number
        self.n_vertex = n_vertex
        self.n_area = n_area
        self.n_demand = n_demand
        self.n_edge = n_edge
        self.n_vehicle = n_vehicle
        self.n_electric_vehicle = n_electric_vehicle
        self.n_charging_station = n_charging_station

        # network
        self.vertices = sumo_gym.utils.fmp_utils.cluster_as_area(vertices, n_area)
        self.vertex_idx_area_mapping = {i: v.area for i, v in enumerate(self.vertices)}
        self.demands = demands
        self.edges = edges

        # vehicles
        self.electric_vehicles = electric_vehicles
        self.ev_name_idx_mapping = {
            f"v{i}": i for i, _ in enumerate(self.electric_vehicles)
        }
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
        # `self.ev_name_idx_mapping` is a mapping from ev sumo id to idx in `electric_vehicles`
        electric_vehicles, self.ev_name_idx_mapping = convert_raw_electric_vehicles(
            raw_electric_vehicles
        )

        # departure should be defined for all vehicles
        # `self.departures` and `self.actual_departures` are lists of indices in `vertices`
        # self.departures[i] is the starting point of electric_vehicles[i] (the endpoint of the passed in edge)
        # self.actual_depatures[i] is the actual start vertex of electric_vehicles[i] (the starting point of the passed in edge)
        departures, actual_departures = convert_raw_departures(
            raw_departures,
            self.ev_name_idx_mapping,
            edges,
            self.edge_dict,
            len(electric_vehicles),
        )

        # `demand` is a list of Demand instances
        demands = convert_raw_demand(raw_demand, self.vertex_dict)

        # set the FMP variables
        self.vertices = vertices
        self.edges = edges
        self.charging_stations = charging_stations
        self.electric_vehicles = electric_vehicles
        self.departures = departures
        self.departures = [int(x) for x in self.departures]
        self.actual_departures = actual_departures
        self.demands = demands

        self.n_demand = len(demands)
        self.n_vertex = len(self.vertices)
        self.n_edge = len(self.edges)
        self.n_vehicle = self.n_electric_vehicle = len(self.electric_vehicles)
        self.n_charging_station = len(self.charging_stations)

    def _is_valid(self):
        if (
            not self.n_vertex
            or not self.n_demand
            or not self.n_edge
            or not self.n_vehicle
            or not self.n_charging_station
            or self.vertices is None
            or self.charging_stations is None
            or self.electric_vehicles is None
            or self.demands is None
            or self.edges is None
            or self.departures is None
        ):
            return False
        if len(self.vertices) != len(self.vertices):
            return False
        if len(self.edges) != len(self.edges):
            return False
        if len(self.electric_vehicles) != len(self.electric_vehicles):
            return False
        if len(self.charging_stations) != len(self.charging_stations):
            return False
        charging_station_locations = set([cs.location for cs in self.charging_stations])
        for d in self.demands:
            if (
                d.departure in charging_station_locations
                or d.destination in charging_station_locations
            ):
                return False
        return True


class FMPActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        self.n = int(n)
        super(FMPActionSpace, self).__init__(n)

    def sample(self) -> int:
        p_to_respond = random.uniform(0.3, 0.6)
        p_to_charge = 1.0 - p_to_respond
        return random.choices([0, 1, 2], [p_to_respond, p_to_charge, 0.0])[0]


class FMPLowerDemandActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        self.n_demand = int(n)
        super(FMPLowerDemandActionSpace, self).__init__(n)

    def sample(self) -> int:
        return random.randint(0, self.n_demand - 1)


class FMPLowerCSActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        self.n_cs = int(n)
        super(FMPLowerCSActionSpace, self).__init__(n)

    def sample(self) -> int:
        return random.randint(0, self.n_cs - 1)


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

        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]
            del kwargs["verbose"]
        else:
            self.verbose = False

        # setup FMP variables
        self._fmp = FMP(**kwargs)
        self.sumo = (
            SumoRender(
                self.sumo_gui_path,
                self.sumo_config_path,
                self.fmp.edge_dict,
                self.fmp.edge_length_dict,
                self.fmp.ev_name_idx_mapping,
                self.fmp.edges,
                self.fmp.vertices,
                self.fmp.vertex_dict,
                self.fmp.n_electric_vehicle,
            )
            if hasattr(self, "render_env") and self.render_env is True
            else None
        )
        self.travel_info = {i: None for i in range(self.fmp.n_electric_vehicle)}

        # setup Petting-Zoo environment variables
        self.possible_agents = list(self.fmp.ev_name_idx_mapping.keys())
        self.agent_name_idx_mapping = self.fmp.ev_name_idx_mapping

        self._action_spaces = {
            agent: FMPActionSpace(3) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: FMPActionSpace(3) for agent in self.possible_agents
        }

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}

        self.upper_rewards = {agent: 0.0 for agent in self.agents}
        self.lower_reward_demand = 0  # network specific, no agent info
        self.lower_reward_cs = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.dones = {agent: False for agent in self.agents}
        self.infos = Metrics()

        self.states = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return FMPActionSpace(3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return FMPActionSpace(3)

    def observe(self, agent):
        return self.observations[agent]

    def action_space_lower_demand(self):
        return FMPLowerDemandActionSpace(self.fmp.n_demand)

    def action_space_lower_cs(self):
        return FMPLowerCSActionSpace(self.fmp.n_charging_station)

    def reset(self):
        """
        Reset Petting-Zoo environment variables.
        """
        for i, ev in enumerate(self.fmp.electric_vehicles):
            self.fmp.electric_vehicles[i].location = self.fmp.departures[i]
            self.fmp.electric_vehicles[i].battery = ev.capacity
            self.fmp.electric_vehicles[i].status = 0
            self.fmp.electric_vehicles[i].responded = list()
        for i, cs in enumerate(self.fmp.charging_stations):
            self.fmp.charging_stations[i].n_slot = cs.n_slot or 1
            self.fmp.charging_stations[i].charging_vehicle = (
                cs.charging_vehicle or list()
            )

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.upper_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.lower_reward_demand = 0
        self.lower_reward_cs = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}

        self.dones = {agent: False for agent in self.possible_agents}
        self.infos = Metrics()

        self.states = {
            agent: get_safe_indicator(
                self.fmp.vertices,
                self.fmp.edges,
                self.fmp.demands,
                self.fmp.charging_stations,
                self.fmp.electric_vehicles[i].location,
                self.fmp.electric_vehicles[i].battery,
            )
            for i, agent in enumerate(self.possible_agents)
        }
        self.observations = {agent: self.states[agent] for agent in self.agents}

        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.travel_info = {i: None for i in range(self.fmp.n_electric_vehicle)}

        return self.observations

    def _was_done_step(self, action):
        if action is not None:
            raise ValueError("when an agent is done, the only valid action is None")

        # removes done agent
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]
        assert self.dones[
            agent
        ], "an agent that was not done as attempted to be removed"
        del self.dones[agent]
        del self.upper_rewards[agent]
        del self._cumulative_rewards[agent]
        self.agents.remove(agent)

        # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

        if self.fmp.electric_vehicles[agent_idx].status > 2 * self.fmp.n_demand:
            if (
                self.fmp.electric_vehicles[agent_idx].status
                > 2 * self.fmp.n_demand + self.fmp.n_charging_station
            ):
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - self.fmp.n_charging_station
                    - 1
                )
            else:
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - 1
                )

            if agent in self.fmp.charging_stations[cs_idx].charging_vehicle:
                self.fmp.charging_stations[cs_idx].charging_vehicle.remove(agent)

    def step(self, action):
        """
        Step takes an action for the current agent.
        Update of state, normal reward, responded, observation are in state "move" and "transition".
        Update of done, broken reward, broken responded and agent are in step.
        """

        upper_action, lower_action = action

        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]

        need_action = (
            self.sumo.retrieve_need_action_status()[agent_idx] if self.sumo else True
        )
        stopped = {
            self.sumo.retrieve_stop_status()[agent_idx] if self.sumo else None
        }

        self.travel_info[agent_idx] = None
        prev_loc = self.fmp.electric_vehicles[agent_idx].location

        if self.dones[agent]:
            self._was_done_step(None)
            self.observations[agent] = None
            self.agent_selection = (
                self._agent_selector.next()
                if self._agent_selector.agent_order
                else None
            )
            self.travel_info[agent_idx] = (prev_loc, IDLE_LOCATION)
            return self.observations, self.upper_rewards, self.dones, self.infos
        else:
            if not need_action:
                print("... Agent: ", agent, " still on the edge...")

            else:
                self.upper_rewards[agent] = 0
                # state move
                if self.fmp.electric_vehicles[agent_idx].status != 0:
                    self._state_move()
                # state transition
                else:
                    self._state_transition(upper_action, lower_action)

                if prev_loc != self.fmp.electric_vehicles[agent_idx].location:
                    self.travel_info[agent_idx] = (
                        prev_loc,
                        self.fmp.electric_vehicles[agent_idx].location,
                    )

                # judge whether it's done
                self.dones[agent] = (
                    set(
                        chain.from_iterable(
                            [ev.responded for ev in self.fmp.electric_vehicles]
                        )
                    )
                    == set(range(self.fmp.n_demand))
                    and self.fmp.electric_vehicles[agent_idx].status == 0
                )

                # check whether is in negative battery
                if self.fmp.electric_vehicles[agent_idx].battery < 0:
                    self.dones[agent] = True
                    if (
                        0
                        < self.fmp.electric_vehicles[agent_idx].status
                        <= 2 * self.fmp.n_demand
                    ):
                        dmd_idx = (
                            self.fmp.electric_vehicles[agent_idx].status
                            - self.fmp.n_demand
                            - 1
                            if self.fmp.electric_vehicles[agent_idx].status
                            > self.fmp.n_demand
                            else self.fmp.electric_vehicles[agent_idx].status - 1
                        )
                        del self.fmp.electric_vehicles[agent_idx].responded[
                            len(self.fmp.electric_vehicles[agent_idx].responded)
                            - self.fmp.electric_vehicles[agent_idx]
                            .responded[::-1]
                            .index(dmd_idx)
                            - 1
                        ]

                self._cumulative_rewards[agent] += self.upper_rewards[agent]

                if self.verbose:
                    print(
                        f"Obs: {self.observations[agent]}; "
                        + f"Rew: {self.upper_rewards[agent]}; "
                        + f"Cum_rew: {self._cumulative_rewards[agent]}; "
                        + f"EV: {self.fmp.electric_vehicles[agent_idx]}."
                    )
            if self._agent_selector.is_last():
                self.num_moves += 1

                self.infos.task_finish_time = self.num_moves
                self.infos.respond_failing_time += self.fmp.n_demand - len(
                    set(
                        chain.from_iterable(
                            [ev.responded for ev in self.fmp.electric_vehicles]
                        )
                    )
                )
                self.infos.total_battery_consume += len(self.agents)

                if self.sumo is not None:
                    self.sumo.update_travel_vertex_info_for_vehicle(self.travel_info)
                    self.render()

                if self.verbose:
                    print("------------------------------")

            self.agent_selection = self._agent_selector.next()

            return self.observations, self.upper_rewards, self.dones, self.infos

    def _state_move(self):
        """
        Deal with moving state:
        1. Move the state
        2. Update observation (force to change observations' action) and reward, accordingly
        """
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]

        # if responding
        if 0 < self.fmp.electric_vehicles[agent_idx].status <= 2 * self.fmp.n_demand:
            if self.fmp.electric_vehicles[agent_idx].status > self.fmp.n_demand:
                dmd_idx = (
                    self.fmp.electric_vehicles[agent_idx].status - self.fmp.n_demand - 1
                )
                dest_loc = self.fmp.demands[dmd_idx].destination
                if self.verbose:
                    print("Move: ", agent, " is in responding demand ", dmd_idx)

                self.fmp.electric_vehicles[
                    agent_idx
                ].location = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.electric_vehicles[agent_idx].location,
                    dest_loc,
                )
                self.fmp.electric_vehicles[agent_idx].battery -= 1
                if self.fmp.electric_vehicles[agent_idx].location == dest_loc:
                    self.fmp.electric_vehicles[agent_idx].status = 0

            else:
                dmd_idx = self.fmp.electric_vehicles[agent_idx].status - 1
                dest_loc = self.fmp.demands[dmd_idx].departure
                if self.verbose:
                    print("Move: ", agent, " is to respond demand ", dmd_idx)

                self.fmp.electric_vehicles[
                    agent_idx
                ].location = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.electric_vehicles[agent_idx].location,
                    dest_loc,
                )
                self.fmp.electric_vehicles[agent_idx].battery -= 1
                if self.fmp.electric_vehicles[agent_idx].location == dest_loc:
                    self.fmp.electric_vehicles[agent_idx].status += self.fmp.n_demand

        # if charging
        elif self.fmp.electric_vehicles[agent_idx].status > 2 * self.fmp.n_demand:
            if (
                self.fmp.electric_vehicles[agent_idx].status
                > 2 * self.fmp.n_demand + self.fmp.n_charging_station
            ):
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - self.fmp.n_charging_station
                    - 1
                )
                if self.verbose:
                    print(
                        "Move: ", agent, " is in charging at ", cs_idx,
                        " with position: ", self.fmp.charging_stations[cs_idx].charging_vehicle.index(agent), 
                        ". Total vehicles in charging station: ", self.fmp.charging_stations[cs_idx].charging_vehicle
                    )

                cs_edge_index = next(i for i in range(len(self.fmp.edges)) if self.fmp.edges[i].end == self.fmp.charging_stations[cs_idx].location)
                cs_lane_position = self.fmp.edge_length_dict[cs_edge_index]
                print("CHECKKK: ", cs_lane_position)

                if (
                    self.fmp.charging_stations[cs_idx].charging_vehicle.index(agent)
                    < self.fmp.charging_stations[cs_idx].n_slot
                    or
                    (stopped is not None and abs(stopped - cs_lane_position) < 5)
                ):
                    self.fmp.electric_vehicles[agent_idx].battery = min(
                        self.fmp.electric_vehicles[agent_idx].battery
                        + self.fmp.charging_stations[cs_idx].charging_speed,
                        self.fmp.electric_vehicles[agent_idx].capacity,
                    )
                else:
                    self.infos.charge_waiting_time += 1

                if (
                    self.fmp.electric_vehicles[agent_idx].battery
                    >= self.fmp.electric_vehicles[
                        self.agent_name_idx_mapping[agent]
                    ].capacity
                ):
                    self.fmp.electric_vehicles[agent_idx].status = 0
                    self.fmp.charging_stations[cs_idx].charging_vehicle.remove(agent)
                    print("Charging finished for vehicle; ", agent, self.fmp.charging_stations[cs_idx].charging_vehicle)
            else:
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - 1
                )
                dest_loc = self.fmp.charging_stations[cs_idx].location
                if self.verbose:
                    print("Move: ", agent, "is to go to charge at ", cs_idx)
                self.fmp.electric_vehicles[
                    agent_idx
                ].location = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.electric_vehicles[agent_idx].location,
                    dest_loc,
                )
                self.fmp.electric_vehicles[agent_idx].battery -= 1

                if self.fmp.electric_vehicles[agent_idx].location == dest_loc:
                    self.fmp.electric_vehicles[
                        agent_idx
                    ].status += self.fmp.n_charging_station
                    self.fmp.charging_stations[cs_idx].charging_vehicle.append(agent)

        # not charging and not loading should not be moving
        else:
            raise ValueError("Agent that not responding or charging should not move")

        self.states[agent] = 3
        self.observations[agent] = 3
        self.upper_rewards[agent] = 0

    def _state_transition(self, upper_action, lower_action):
        """
        Transit the state of current agent according to the action and update observation:
        1. Update states (if "move" action for a state need to make decision, then no change)
        2. Update responded if responding a new demand
        3. Update observation and reward, accordingly
        """
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]
        is_valid = 1

        if upper_action == 2:
            if self.verbose:
                print("Trans: ", agent, "is taking moving action")

        # action to charge
        elif upper_action == 1:
            if self.verbose:
                print("Trans: ", agent, "is to go to charge at ", lower_action)

            self.fmp.electric_vehicles[agent_idx].location = one_step_to_destination(
                self.fmp.vertices,
                self.fmp.edges,
                self.fmp.electric_vehicles[agent_idx].location,
                self.fmp.charging_stations[lower_action].location,
            )
            self.fmp.electric_vehicles[agent_idx].battery -= 1
            self.fmp.electric_vehicles[agent_idx].status = (
                2 * self.fmp.n_demand + lower_action + 1
            )

        # action to load
        elif upper_action == 0:
            if self.verbose:
                print(
                    "Trans: ",
                    agent,
                    " is to respond demand ",
                    lower_action,
                )

            self.fmp.electric_vehicles[agent_idx].location = one_step_to_destination(
                self.fmp.vertices,
                self.fmp.edges,
                self.fmp.electric_vehicles[agent_idx].location,
                self.fmp.demands[lower_action].departure,
            )
            self.fmp.electric_vehicles[agent_idx].battery -= 1
            self.fmp.electric_vehicles[agent_idx].status = 1 + lower_action
            is_valid = (
                1
                if lower_action
                not in set(
                    chain.from_iterable(
                        [ev.responded for ev in self.fmp.electric_vehicles]
                    )
                )
                else 0
            )
            self.fmp.electric_vehicles[agent_idx].responded.append(lower_action)

        self.states[agent] = get_safe_indicator(
            self.fmp.vertices,
            self.fmp.edges,
            self.fmp.demands,
            self.fmp.charging_stations,
            self.fmp.electric_vehicles[agent_idx].location,
            self.fmp.electric_vehicles[agent_idx].battery,
        )
        self.observations[agent] = self.states[agent]
        self._calculate_upper_reward(agent, agent_idx, upper_action, lower_action)
        # self._calculate_lower_reward(self.fmp.electric_vehicles[agent_idx].location, is_valid, upper_action, lower_action)

    def _calculate_upper_reward(self, agent, agent_idx, upper_action, lower_action):
        self.upper_rewards[agent] = 0
        if self.states[agent] == 0:
            if upper_action == 0:
                self.upper_rewards[agent] = -100
            elif upper_action == 1:
                self.upper_rewards[agent] = 100
        elif self.states[agent] == 1:
            if upper_action == 0:
                next_state = get_safe_indicator(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.demands,
                    self.fmp.charging_stations,
                    self.fmp.demands[lower_action].destination,
                    self.fmp.electric_vehicles[agent_idx].battery
                    - get_dist_to_finish_demands(
                        self.fmp.vertices,
                        self.fmp.edges,
                        self.fmp.demands,
                        self.fmp.electric_vehicles[agent_idx].location,
                    )[lower_action],
                )
                self.upper_rewards[agent] = -100 if next_state == 0 else 50
            elif upper_action == 1:
                self.upper_rewards[agent] = 20
        elif self.states[agent] == 2:
            if upper_action == 0:
                self.upper_rewards[agent] = 50
            elif upper_action == 1:
                self.upper_rewards[agent] = -20

    def last(self, observe=True):
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]
        if agent is None:
            return None, 0, True, {}
        observation = self.observe(agent) if observe else None

        loc_area = self.fmp.vertices[
            self.fmp.electric_vehicles[agent_idx].location
        ].area
        demand_vector = self._generate_demand_vector()
        demand_vector.append(loc_area)
        cs_vector = self._generate_cs_vector()
        cs_vector.append(loc_area)
        return (
            observation,
            self.upper_rewards[agent],
            self.dones[agent],
            self.infos,
        ), (
            (demand_vector, cs_vector),
            (self.lower_reward_demand, self.lower_reward_cs),
            self.dones[agent],
            self.infos,
        )

    def _generate_cs_vector(self):
        return [len(cs.charging_vehicle) for cs in self.fmp.charging_stations]

    def _generate_demand_vector(self):
        responded_list = set(
            chain.from_iterable([ev.responded for ev in self.fmp.electric_vehicles])
        )
        all_demand = list(range(self.fmp.n_demand))

        return [1 if d in responded_list else 0 for d in all_demand]

    def render(self, mode="human"):
        if self.sumo_gui_path is None:
            raise EnvironmentError("Need sumo-gui path to render")
        elif self.sumo is not None:
            self.sumo.render()

    def close(self):
        if hasattr(self, "sumo") and self.sumo is not None:
            self.sumo.close()
