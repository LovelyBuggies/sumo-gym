import sumo_gym.utils.network_utils as network_utils
import numpy as np
import numpy.typing as npt
from typing import Tuple

NO_LOADING = -1
NO_CHARGING = -1
CHARGING_STATION_LENGTH = 5


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

    def __repr__(self):
        return f"ElectricVehicles ({self.id}, {self.speed}, {self.indicator}, {self.capacity})"


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

    def __repr__(self):
        return f"ChargingStation ({self.location}, {self.indicator}, {self.charging_speed})"


class Loading(object):
    def __init__(self, current=-1, target=-1):
        self.current = current
        self.target = target

    def __repr__(self):
        return f"(responding {self.current}, goto respond {self.target})"


class Charging(object):
    def __init__(self, charging_station=-1, battery_charged=0):
        self.charging_station = charging_station
        self.battery_charged = battery_charged

    def __repr__(self):
        return f"(goto charge {self.charging_station}, battery added {self.battery_charged})"


class GridAction(object):
    def __init__(self, state=None):
        self.is_loading = state.is_loading
        self.is_charging = Charging(state.is_charging, 0)
        self.location = state.location

    def __repr__(self):
        return f"({self.is_loading}, {self.is_charging}, location {self.location})"


def convert_raw_vertices(raw_vertices):
    """
    Each raw vertex is [id (str), x_coord (float), y_coord (float)]
    """
    vertices = []
    vertex_dict = {}  # vertex id in SUMO to idx in vertices
    for counter, v in enumerate(raw_vertices):
        vertices.append(Vertex(v[1], v[2]))
        vertex_dict[v[0]] = counter
    return vertices, vertex_dict


def convert_raw_edges(raw_edges, vertex_dict):
    """
    Each raw edge is
    [id (str), from_vertex_id (str), to_vertex_id (str), edge_length (float)]
    """
    edges = []
    edge_dict = {}  # sumo edge_id to idx in edges
    edge_length_dict = {}  # sumo edge_id to length
    for counter, e in enumerate(raw_edges):
        new_edge = Edge(vertex_dict[e[1]], vertex_dict[e[2]])
        edges.append(new_edge)
        edge_dict[e[0]] = counter
        edge_length_dict[e[0]] = e[3]
    return edges, edge_dict, edge_length_dict


def euclidean_distance(start_x, start_y, end_x, end_y):
    """
    Compute euclidean distance between (start_x, start_y)
    and (end_x, end_y)
    """
    return (((start_x - end_x) ** 2) + ((start_y - end_y) ** 2)) ** 0.5


def convert_raw_charging_stations(
    raw_charging_stations, vertices, edges, edge_dict, edge_length_dict
):
    """
    Each raw charging station is
    [id, (x_coord, y_coord), edge_id, charging speed]
    """

    charging_station_dict = {}  # idx in charging_stations to sumo id
    charging_stations = []

    vtx_counter = len(vertices)

    for counter, charging_station in enumerate(raw_charging_stations):

        charging_station_dict[counter] = charging_station[0]

        # create new vertex with charging station's location
        x_coord, y_coord = charging_station[1]
        new_vtx = Vertex(x_coord, y_coord)
        vertices.append(new_vtx)

        # create two new edges
        # first get the start and end vertex indices of the old edge
        edge_id = charging_station[2]
        old_edge_start_idx = edges[edge_dict[edge_id]].start
        old_edge_end_idx = edges[edge_dict[edge_id]].end

        edge_length_positive_edge_cs = charging_station[4]
        edge_length_postive_edge = edge_length_dict[edge_id]

        curr_edge_count = len(edge_dict)
        edges.append(Edge(old_edge_start_idx, vtx_counter))
        edge_dict["split1_%s" % edge_id] = curr_edge_count
        edge_length_dict["split1_%s" % edge_id] = edge_length_positive_edge_cs

        curr_edge_count += 1
        edges.append(Edge(vtx_counter, old_edge_start_idx))
        edge_dict["split1_-%s" % edge_id] = curr_edge_count
        edge_length_dict["split1_-%s" % edge_id] = edge_length_positive_edge_cs

        curr_edge_count += 1
        edges.append(Edge(vtx_counter, old_edge_end_idx))
        edge_dict["split2_%s" % edge_id] = curr_edge_count
        edge_length_dict["split2_%s" % edge_id] = (
            edge_length_postive_edge
            - edge_length_positive_edge_cs
            + CHARGING_STATION_LENGTH
        )

        curr_edge_count += 1
        edges.append(Edge(old_edge_end_idx, vtx_counter))
        edge_dict["split2_-%s" % edge_id] = curr_edge_count
        edge_length_dict["split2_-%s" % edge_id] = (
            edge_length_postive_edge
            - edge_length_positive_edge_cs
            + CHARGING_STATION_LENGTH
        )

        # instantiate new ChargingStation with location set to idx in `vertices`
        charging_stations.append(ChargingStation(vtx_counter, 220, charging_station[3]))

        vtx_counter += 1

    return charging_stations, charging_station_dict, edge_length_dict


def convert_raw_electric_vehicles(raw_electric_vehicles):
    """
    Each raw electric vehicle is
    [id (str), maximum speed (float), maximumBatteryCapacity (float)]
    """

    electric_vehicles = []
    ev_dict = {}  # ev sumo id to idx in electric_vehicles
    for counter, vehicle in enumerate(raw_electric_vehicles):
        electric_vehicles.append(ElectricVehicles(counter, vehicle[1], 220, vehicle[2]))
        ev_dict[vehicle[0]] = counter

    return electric_vehicles, ev_dict


def convert_raw_departures(raw_departures, ev_dict, edges, edge_dict, num_vehicles):
    """
    Each raw departure is [vehicle_id, starting_edge_id]
    """
    departures = np.zeros(num_vehicles)
    actual_departures = np.zeros(num_vehicles)
    for dpt in raw_departures:
        actual_departures[ev_dict[dpt[0]]] = edges[edge_dict[dpt[1]]].start
        departures[ev_dict[dpt[0]]] = edges[edge_dict[dpt[1]]].end
    return departures, actual_departures


def convert_raw_demand(raw_demand, vertex_dict):
    """
    Each raw demand is [junction_id, dest_vertex_id]
    """
    demand = []
    for d in raw_demand:
        demand.append(Demand(vertex_dict[d[0]], vertex_dict[d[1]]))
    return demand


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
