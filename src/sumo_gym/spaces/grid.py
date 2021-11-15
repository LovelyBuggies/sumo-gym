from typing import Tuple
import random

import sumo_gym
import sumo_gym.utils.grid_utils as grid_utils
import gym
from sumo_gym.typing import FMPElectricVehiclesType, FMPDemandsType, FMPChargingStationType

import numpy as np
import numpy.typing as npt

class GridSpace(gym.spaces.Space):
    def __init__(
        self,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.typing.FMPDemandsType = None,
        responded: set = None,
        edges: sumo_gym.typing.EdgeType = None,
        electric_vehicles: FMPElectricVehiclesType = None,
        charging_stations: sumo_gym.typing.FMPChargingStationType = None,
        locations: sumo_gym.typing.LocationsType = None,
        batteries: npt.NDArray[float] = None,
        is_loading: npt.NDArray[int] = None,
        is_charging: npt.NDArray[int] = None,
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
        self.locations = locations
        self.batteries = batteries
        self.is_loading = is_loading
        self.is_charging = is_charging

    def sample(self) -> npt.NDArray[int]: # returned samples' ele (is_loading, is_charing, the one-step loc)
        n_vehicle = len(self.is_loading)
        samples = [(-1, -1, 0) for i in range(n_vehicle)]
        for i in range(n_vehicle):
            if self.is_loading[i][0] != -1: # is on the way
                loc = grid_utils.one_step_to_destination(self.vertices, self.edges, self.locations[i], self.demand[self.is_loading[i][0]][1])
                print("----- In the way of demand:", self.is_loading[i][0])
                samples[i] = ((-1, -1), -1, loc) if loc == self.demand[self.is_loading[i][0]][1] else ((self.is_loading[i][0], self.is_loading[i][1]), -1, loc)
            elif self.is_loading[i][1] != -1: # is to the way
                loc = grid_utils.one_step_to_destination(self.vertices, self.edges, self.locations[i], self.demand[self.is_loading[i][1]][0])
                print("----- In the way to respond:", self.is_loading[i][1])
                samples[i] = ((self.is_loading[i][1], self.is_loading[i][1]), -1, loc) if loc == self.demand[self.is_loading[i][1]][0] else ((self.is_loading[i][0], self.is_loading[i][1]), -1, loc)
            elif self.is_charging[i] != -1: # is charging
                if self.electric_vehicles[i][3] - self.batteries[i] > self.charging_stations[self.is_charging[i]][2]:
                    print("----- Still charging")
                    samples[i] = ((-1, -1), self.is_charging[i], self.charging_stations[self.is_charging[i]][0])
                else:
                    print("----- Charging finished")
                    samples[i] = ((-1, -1), -1, self.charging_stations[self.is_charging[i]][0])
            else: # available
                ncs, _ = grid_utils.nearest_charging_station_with_distance(self.vertices, self.charging_stations, self.edges, self.locations[i])
                diagonal_len = 2 * (max(self.vertices, key=lambda item:item[1])[1] - min(self.vertices, key=lambda item:item[1])[1] \
                               + max(self.vertices, key=lambda item:item[0])[0] - min(self.vertices, key=lambda item:item[0])[0])
                possibility_of_togo_charge = self.batteries[i] / (diagonal_len - self.electric_vehicles[i][3]) \
                                + self.electric_vehicles[i][3] / (self.electric_vehicles[i][3] - diagonal_len)
                if np.random.random() < possibility_of_togo_charge:
                    loc = grid_utils.one_step_to_destination(self.vertices, self.edges, self.locations[i], self.charging_stations[ncs][0])
                    print("----- Goto charge:", ncs)
                    samples[i] = ((-1, -1), ncs, loc) if loc == self.charging_stations[ncs][0] else ((-1, -1), -1, loc)
                else:
                    available_dmd = [d for d in range(len(self.demand)) if d not in self.responded]
                    if len(available_dmd):
                        dmd_idx = random.choices(available_dmd)[0]
                        print("----- Choose dmd_idx:", dmd_idx)
                        self.responded.add(dmd_idx)
                        loc = grid_utils.one_step_to_destination(self.vertices, self.edges, self.locations[i], self.demand[dmd_idx][0])
                        samples[i] = ((dmd_idx, dmd_idx), -1, loc) if loc == self.demand[dmd_idx][0] else ((-1, dmd_idx), -1, loc)
                    else:
                        samples[i] = ((-1, -1), -1, self.locations[i])

        print("Samples: ", samples)
        return samples