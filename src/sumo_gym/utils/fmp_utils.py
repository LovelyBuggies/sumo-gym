import sumo_gym.utils.network_utils as network_utils
import numpy as np
import numpy.typing as npt
from typing import Tuple

NO_LOADING = -1
NO_CHARGING = -1


class Vertex(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"vertex ({self.x}, {self.y})"


class Edge(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other):
        return (self.start, self.end) < (other.start, other.end)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"edge ({self.start}, {self.end})"


class Demand(object):
    def __init__(self, departure, destination):
        self.departure = departure
        self.destination = destination

    def __eq__(self, other):
        return (self.departure, self.destination) == (
            other.departure,
            other.destination,
        )

    def __lt__(self, other):
        return (self.departure, self.destination) < (other.departure, other.destination)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"demand ({self.departure}, {self.destination})"


class ElectricVehicles(object):
    def __init__(self, id, speed, indicator, capacity):
        self.id = id
        self.speed = speed
        self.indicator = indicator
        self.capacity = capacity

    def __eq__(self, other):
        return self.location == other.location

    def __lt__(self, other):
        return self.location < other.location

    def __hash__(self):
        return hash(str(self))


class ChargingStation(object):
    def __init__(self, location, indicator, charging_speed):
        self.location = location
        self.indicator = indicator
        self.charging_speed = charging_speed

    def __eq__(self, other):
        return self.location == other.location

    def __lt__(self, other):
        return self.location < other.location

    def __hash__(self):
        return hash(str(self))


class Loading(object):
    def __init__(self, current=-1, target=-1):
        self.current = current
        self.target = target

    def __repr__(self):
        return f"(responding {self.current}, goto respond {self.target})"


class GridAction(object):
    def __init__(self, state=None):
        self.is_loading = state.is_loading
        self.is_charging = state.is_charging
        self.location = state.location

    def __repr__(self):
        return f"({self.is_loading}, goto charge {self.is_charging}, location {self.location})"


def one_step_to_destination(vertices, edges, start_index, dest_index):
    if start_index == dest_index:
        return dest_index
    visited = [False] * len(vertices)
    bfs_queue = [dest_index]
    visited[dest_index] = True

    while bfs_queue:
        curr = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v == start_index:
                return curr
            elif not visited[v]:
                bfs_queue.append(v)
                visited[v] = False


def nearest_charging_station_with_distance(
    vertices, charging_stations, edges, start_index
):
    charging_station_vertices = [
        charging_station.location for charging_station in charging_stations
    ]
    visited = [False] * len(vertices)

    bfs_queue = [[start_index, 0]]
    visited[start_index] = True

    while bfs_queue:
        curr, curr_depth = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v in charging_station_vertices:
                return charging_station_vertices.index(v), curr_depth + 1
            elif not visited[v]:
                bfs_queue.append([v, curr_depth + 1])
                visited[v] = False


def dist_between(vertices, edges, start_index, dest_index):
    if start_index == dest_index:
        return 0
    visited = [False] * len(vertices)
    bfs_queue = [[start_index, 0]]
    visited[start_index] = True
    while bfs_queue:
        curr, curr_depth = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v == dest_index:
                return curr_depth + 1
            elif not visited[v]:
                bfs_queue.append([v, curr_depth + 1])
                visited[v] = False


def get_hot_spot_weight(vertices, edges, demands, demand_start):
    adjacent_vertices = np.append(
        network_utils.get_adj_list(vertices, edges)[demand_start], demand_start
    )
    local_demands = len([d for d in demands if d.departure in adjacent_vertices])

    return local_demands / len(demands) * 100
