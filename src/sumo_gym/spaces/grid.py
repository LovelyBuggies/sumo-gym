from typing import Tuple
import random
import sumo_gym.typing
import sumo_gym.utils.grid_utils as grid_utils


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
        is_loading: npt.NDArray[bool] = None,
        shape=None,
        dtype=None,
        seed=None,
    ):
        super(NetworkSpace, self).__init__()
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

    def sample(self) -> npt.NDArray[bool, int]: # returned samples' ele (is_loading, is_charing, the one-step loc)
        n_vehicle = len(self.is_loading)
        samples = np.zeros(n_vehicle)
        for i in range(n_vehicle):
            if self.is_loading[i] != -1:
                loc = grid_utils.one_step_to_destination(self.vertices, self.edges, self.locations[i], self.demand[self.is_loading][1])
                samples[i] = (-1, -1, loc) if loc == self.demand[self.is_loading][1] else (self.is_loading, -1, loc)
            else:
                if self.is_charging[i] != -1:
                    if self.electric_vehicles[i][3] - self.batteries[i] > speed:
                        samples[i] = (-1, self.is_charging[i], self.charging_stations[self.is_charging[i]])
                    else:
                        samples[i] = (-1, -1, self.charging_stations[self.is_charging[i]])
                else:
                    ncs, battery_threshold = grid_utils.nearest_charging_station_with_distance(self.vertices, self.charging_stations, self.edges, self.locations[i])  # one step towards
                    possibility_of_togo_charge = -(self.batteries[i] - battery_threshold) / (self.electric_vehicles[i][3] - battery_threshold) + 1
                    if np.random.random() < possibility_of_togo_charge:
                        loc = grid_utils.one_step_to_destination(self.vertices, self.edges, self.locations[i], self.charging_stations[ncs])
                        samples[i] = (-1, ncs, loc) if loc == self.charging_stations[ncs] else (-1, -1, loc)
                    else:
                        dmd_idx = random.sample(set(range(len(self.demand))) - self.responded, 1)[0]
                        loc = grid_utils.one_step_to_destination(self.locations[i], self.demand[dmd_idx])
                        samples[i] = (dmd_idx, -1, loc) if loc == self.demand[dmd_idx] else (-1, -1, loc)

        return samples