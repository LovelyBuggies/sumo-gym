from typing import Tuple

import sumo_gym.typing


class GridSpace(gym.spaces.Space):
    def __init__(
        self,
        vertices: sumo_gym.typing.VerticesType = None,
        demand: sumo_gym.typing.FMPDemandsType = None,
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
        self.edges = edges
        self.electric_vehicles = electric_vehicles
        self.charging_stations = charging_stations
        self.locations = locations
        self.batteries = batteries
        self.is_loading = is_loading
        self.is_charging = is_charging

    def sample(self) -> npt.NDArray[bool, int]: # is_loading, is_charing, the one-step loc
        n_vehicle = len(self.is_loading)
        samples = np.zeros(n_vehicle)
        for i in range(n_vehicle):
            if self.is_loading[i] != -1:
                loc = self.demand[self.is_loading][1]  # one step towards this direction
                samples[i] = (-1, -1, loc) if loc == self.demand[self.is_loading][1] else (self.is_loading, -1, loc)
            else:
                if self.is_charging[i] != -1:
                    if self.electric_vehicles[i][2] - self.batteries[i] > speed:
                        samples[i] = (-1, self.is_charging[i], self.charging_stations[self.is_charging[i]])
                    else:
                        samples[i] = (-1, -1, self.charging_stations[self.is_charging[i]])
                else:
                    ncs, battery_threshold = find_the_nearest_charging_station_and_its_distance()  # one step towards
                    possibility_of_togo_charge = -log(self.batteries[i] - battery_threshold)
                    if self.batteries[i] < battery_threshold or np.random.random() < possibility_of_togo_charge:
                        loc = self.charging_stations[ncs] # one step towards this direction
                        samples[i] = (-1, ncs, loc) if loc == self.charging_stations[ncs] else (-1, -1, loc)
                    else:
                        dmd_idx = find_a_request_to_respond()
                        loc = self.demand[dmd_idx]  # one step towards this direction
                        samples[i] = (dmd_idx, -1, loc) if loc == self.demand[dmd_idx] else (-1, -1, loc)

        return samples