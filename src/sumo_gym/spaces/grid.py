from typing import Any
import random

import sumo_gym
from sumo_gym.utils.fmp_utils import Loading, GridAction, NO_LOADING, NO_CHARGING
import gym

import numpy as np
import numpy.typing as npt

from sumo_gym.utils.sumo_utils import SumoRender


class GridSpace(gym.spaces.Space):
    def __init__(
        self,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.utils.fmp_utils.Demand = None,
        responded: set = None,
        edges: sumo_gym.typing.EdgeType = None,
        electric_vehicles: sumo_gym.utils.fmp_utils.ElectricVehicles = None,
        charging_stations: sumo_gym.utils.fmp_utils.ChargingStation = None,
        states=None,
        sumo: SumoRender = None,
        shape=None,
        dtype=None,
        seed=None,
    ):
        super(GridSpace, self).__init__()
        self.vertices = vertices
        self.demand = demand
        self.responded = responded
        self.edges = edges
        self.electric_vehicles = electric_vehicles
        self.charging_stations = charging_stations
        self.states = states
        self.sumo = sumo
        print("     DEMAND: ", demand)

    def sample(
        self,
    ) -> Any:
        n_vehicle = len(self.states)
        samples = [GridAction(state) for state in self.states]

        # extract all demands being responded
        responding = set()
        for i in range(n_vehicle):
            if self.states[i].is_loading.current != NO_LOADING:
                responding.add(self.states[i].is_loading.current)
            elif self.states[i].is_loading.target != NO_LOADING:
                responding.add(self.states[i].is_loading.target)  # todo

        stop_statuses = self.sumo.get_stop_status()
        for i in range(n_vehicle):
            if not stop_statuses[i]:
                print("----- Traveling along the edge...")
                continue

            if self.states[i].is_loading.current != NO_LOADING:  # is on the way
                print("----- In the way of demand:", self.states[i].is_loading.current)
                loc = sumo_gym.utils.fmp_utils.one_step_to_destination(
                    self.vertices,
                    self.edges,
                    self.states[i].location,
                    self.demand[self.states[i].is_loading.current].destination,
                )

                if loc == self.demand[self.states[i].is_loading.current].destination:
                    samples[i].is_loading = Loading(NO_LOADING, NO_LOADING)
                else:
                    samples[i].is_loading = Loading(
                        self.states[i].is_loading.current,
                        self.states[i].is_loading.target,
                    )
                samples[i].location = loc
            elif self.states[i].is_loading.target != NO_LOADING:  # is to the way
                print("----- In the way to respond:", self.states[i].is_loading.target)
                loc = sumo_gym.utils.fmp_utils.one_step_to_destination(
                    self.vertices,
                    self.edges,
                    self.states[i].location,
                    self.demand[self.states[i].is_loading.target].departure,
                )
                samples[i].location = loc
                if loc == self.demand[self.states[i].is_loading.target].departure:
                    samples[i].is_loading = Loading(
                        self.states[i].is_loading.target,
                        self.states[i].is_loading.target,
                    )
                else:
                    samples[i].is_loading = Loading(
                        self.states[i].is_loading.current,
                        self.states[i].is_loading.target,
                    )
            elif self.states[i].is_charging != NO_CHARGING:  # is charging
                samples[i].location = self.charging_stations[
                    self.states[i].is_charging
                ].location
                if (
                    self.electric_vehicles[i].capacity - self.states[i].battery
                    > self.charging_stations[self.states[i].is_charging].charging_speed
                ):
                    print("----- Still charging")
                    samples[i].is_charging = self.states[i].is_charging
                else:
                    print("----- Charging finished")
            else:  # available
                diagonal_len = 2 * (
                    float(max(self.vertices, key=lambda item: item.y).y)
                    - float(min(self.vertices, key=lambda item: item.y).y)
                    + float(max(self.vertices, key=lambda item: item.x).x)
                    - float(min(self.vertices, key=lambda item: item.x).x)
                )
                probability_of_togo_charge = self.states[i].battery / (
                    diagonal_len - self.electric_vehicles[i].capacity
                ) + self.electric_vehicles[i].capacity / (
                    self.electric_vehicles[i].capacity - diagonal_len
                )
                if np.random.random() < probability_of_togo_charge:
                    (
                        ncs,
                        _,
                    ) = sumo_gym.utils.fmp_utils.nearest_charging_station_with_distance(
                        self.vertices,
                        self.charging_stations,
                        self.edges,
                        self.states[i].location,
                    )
                    print("----- Goto charge:", ncs)
                    loc = sumo_gym.utils.fmp_utils.one_step_to_destination(
                        self.vertices,
                        self.edges,
                        self.states[i].location,
                        self.charging_stations[ncs].location,
                    )
                    samples[i].location = loc
                    if loc == self.charging_stations[ncs].location:
                        samples[i].is_charging = ncs
                else:
                    available_dmd = [
                        d
                        for d in range(len(self.demand))
                        if d not in self.responded and d not in responding
                    ]
                    if len(available_dmd):
                        dmd_idx = random.choices(available_dmd)[0]
                        print("----- Choose dmd_idx:", dmd_idx)
                        responding.add(dmd_idx)
                        loc = sumo_gym.utils.fmp_utils.one_step_to_destination(
                            self.vertices,
                            self.edges,
                            self.states[i].location,
                            self.demand[dmd_idx].departure,
                        )
                        samples[i].location = loc
                        if loc == self.demand[dmd_idx].departure:
                            samples[i].is_loading = Loading(dmd_idx, dmd_idx)
                        else:
                            samples[i].is_loading = Loading(NO_LOADING, dmd_idx)
                    else:
                        print("----- IDLE...")
                        samples[i].location = self.states[i].location

        print("Samples: ", samples)
        return samples
