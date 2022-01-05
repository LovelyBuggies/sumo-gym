import operator
import random
import sys
import os
import numpy as np
from typing import Type, Tuple, Dict, Any
import sumo_gym.typing

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sumo_gym
from sumo_gym.utils.sumo_utils import SumoRender
from sumo_gym.utils.svg_uitls import vehicle_marker
from sumo_gym.utils.fmp_utils import *

import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers


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


class FMPEnv(AECEnv):
    metadata = {"render.modes": ["human"]}
    fmp = property(operator.attrgetter("_fmp"))
    __isfrozen = False

    def __init__(self, **kwargs):

        # set up sumo related attributes
        self._setup_sumo_attributes(**kwargs)
        self.num_moves = -1

        # set up AEC related attributes, should not be changed after initialization.
        self.possible_agents = self.fmp.ev_dict.keys()
        self.agent_name_mapping = self.fmp.ev_dict
        self._action_spaces = {agent: gym.spaces.Discrete(self.fmp.n_charging_station + len(self.fmp.demand) + 1) for agent in self.possible_agents}
        self._observation_spaces = {agent: gym.spaces.Box(
            low=np.array([0., 0., 0., 0., 0.]),
            high=np.array([self.fmp.n_vertex, self.fmp.electric_vehicles[0].capacity, 1., 1., 1.]),
            dtype=np.float32
        ) for agent in self.possible_agents}

        self._freeze()

    def _setup_sumo_attributes(self,**kwargs):
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

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(
                "Cannot add new attributes once instance %r is initialized" % self
            )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Box(
            low=np.array([0., 0., 0., 0., 0.]),
            high=np.array([self.fmp.n_vertex,
                           self.fmp.electric_vehicles[0].capacity,
                           2*(len(self.fmp.demand) - len(self.responded)) + 1,
                           2*len(self.fmp.n_charging_station) + 1,
                           1.]),
            dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(self.fmp.n_charging_station + len(self.demand_dict_action_space) + 1)


    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. 
        '''
        return np.array(self.observations[agent])

    def _get_default_obs(self, agent):
        return np.asarray(
            [
                self.states[agent].location,
                self.states[agent].battery,
                1. if self.states[agent].is_loading else 0.,
                1. if self.states[agent].is_charging else 0.,
                1. if self.states[agent].is_loading.target == NO_LOADING and self.states[agent].is_charging.target == NO_CHARGING else 0.,
             ]
        )

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.states = {agent: FMPState() for agent in self.agents}
        self.observations = {agent: self._get_default_obs(agent) for agent in self.agents}

        self.prev_locations = {agent: 0 for agent in self.agents}
        self.prev_is_loading = {agent: NO_LOADING for agent in self.agents}

        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stoepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


        self.responded = set()
        for i in range(self.fmp.n_electric_vehicle):
            self.states[i].location = self.fmp.departures[i]
            self.states[i].battery = self.fmp.electric_vehicles[i].capacity

        self.move_space: sumo_gym.spaces.grid.GridSpace = (
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
        self.demand_dict_action_space = dict()
        for i in range(self.fmp.n_charging_station, self.fmp.n_charging_station + len(self.fmp.demand), 1):
            self.demand_dict_action_space[i] = i - self.fmp.n_charging_station

        self.actions: sumo_gym.typing.ActionsType = None
        self.rewards: sumo_gym.typing.RewardsType = np.zeros(self.fmp.n_vehicle)

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''

        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent, or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent and update action space
        self.states[self.agent_selection] = self._convert_discrete_action_to_move(action)
        self._update_demand_space(action)
        
        self._update_battery_for_agent(agent, self.states[self.agent_selection])
        self.observations[agent] = self._get_obs_from_action(self.states[agent])

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary, and update the observations
            for agent in self.agents:
                self.rewards[agent] += self._calculate_reward(agent)

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.responded == set(range(len(self.fmp.demand))) or self.states[self.agent_selection].battery < 0 for agent in self.agents}
                
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.states[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            # self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()

        # update previous info for agent
        self.prev_locations[agent] = self.states[agent].location
        self.prev_is_loading[agent] = self.states[agent].is_loading


    def _update_demand_space(self, action):
        # when a demand is being responding or responded, remove it from action space for other agents
        if not action < self.fmp.n_charging_station:
            action_space_new_len = self.fmp.n_charging_station + len(self.demand_dict_action_space) - 1
            for i in range(action, action_space_new_len, 1):
                self.demand_dict_action_space[i] = self.demand_dict_action_space[i + 1]
            del self.demand_dict_action_space[action_space_new_len]

    def _convert_discrete_action_to_move(self, action, agent):

        # convert action space action to move space action
        if action < self.fmp.n_charging_station:
            converted_action = self.states[agent]
            converted_action.is_loading, converted_action.is_charging = Loading(NO_LOADING, NO_LOADING), Charging(NO_CHARGING, action)
            converted_action.location = one_step_to_destination(
                self.fmp.vertices, self.fmp.edges, self.states[agent].location,
                self.fmp.charging_stations[self.states[agent].is_charging.target].location
            )
        else:
            demand_idx = self.demand_dict_action_space[action]
            converted_action = self.states[agent]
            converted_action.is_loading, converted_action.is_charging = Loading(NO_LOADING, demand_idx), Charging(NO_CHARGING, NO_CHARGING)
            converted_action.location = one_step_to_destination(
                self.fmp.vertices, self.fmp.edges, self.states[agent].location,
                self.fmp.demand[self.states[agent].is_loading.current].destination,
            )

        return converted_action

    def _update_battery_for_agent(self, agent, action):
        self.states[agent].battery -= sumo_gym.utils.fmp_utils.dist_between(
            self.fmp.vertices,
            self.fmp.edges,
            self.prev_locations[agent]
            if action.location == IDLE_LOCATION
            else action.location,
            self.prev_locations[agent],
        )

        if self.states[agent].is_charging.current != NO_CHARGING:
            self.states[agent].battery += self.fmp.charging_stations[
                self.states[agent].is_charging.current
            ].charging_speed

    def _get_obs_from_action(self, action):
        if action.is_loading.current == NO_LOADING and action.is_loading.target == NO_LOADING:
            is_loading = -1
        elif action.is_loading.current == NO_LOADING:
            is_loading = action.is_loading.target
        else:
            is_loading = 2 * action.is_loading.target
        if action.is_charging.current == NO_CHARGING and action.is_charging.target == NO_CHARGING:
            is_charging = -1
        elif action.is_charging.current == NO_CHARGING:
            is_charging = action.is_charging.target
        else:
            is_charging = 2 * action.is_charging.target
        
        return np.asarray(
            [
                action.location,
                action.battery,
                is_loading,
                is_charging,
                True if is_loading == NO_LOADING and is_charging == NO_CHARGING else False,
            ]
        )

    def _calculate_reward(self, agent):
        reward = 0
        if self.states[agent].battery < 0:
            reward -= 1000
            

        if self.prev_is_loading[agent] != -1 and self.states[agent].is_loading.current == -1:
            self.responded.add(self.prev_is_loading[agent])
            reward += sumo_gym.utils.fmp_utils.get_hot_spot_weight(
                self.fmp.vertices,
                self.fmp.edges,
                self.fmp.demand,
                self.fmp.demand[self.prev_is_loading[agent]].departure,
            ) * sumo_gym.utils.fmp_utils.dist_between(
                self.fmp.vertices,
                self.fmp.edges,
                self.fmp.demand[self.prev_is_loading[agent]].departure,
                self.fmp.demand[self.prev_is_loading[agent]].destination,
            )

        return reward

    def render(self, mode="human"):
        if self.sumo_gui_path is None:
            raise EnvironmentError("Need sumo-gui path to render")
        elif self.sumo is not None:
            # TODO: need sumo render modification here
            # self.sumo.render()
            print("sumo render")

    # TODO: need to add default behavior also
    def close(self):
        if hasattr(self, "sumo") and self.sumo is not None:
            self.sumo.close()
