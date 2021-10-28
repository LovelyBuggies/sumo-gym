import numpy as np
import numpy.typing as npt
from typing import Type, Tuple, Dict, Union, Any


class VRP(object):
    def __init__(self, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures):
        # number
        self.vertex_num: int = vertex_num
        self.depot_num: int = depot_num
        self.edge_num: int = edge_num
        self.vehicle_num: int = vehicle_num

        # network
        self.vertices: Dict[int, npt.NDArray[Tuple[np.float64]]] = vertices
        self.depots: Dict[int, npt.NDArray[Tuple[np.float64]]] = {x: y for x, y in enumerate(list(vertices.values())[0:depot_num])}
        self.demand: Dict[int, np.float64] = demand
        self.edges: npt.NDArray[Tuple[int]] = edges

        # vehicles
        self.departures: Dict[int, int] = departures

    def __repr__(self):
        return f"Vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures}.\n"


class VRPState(VRP):
    def __init__(self, id, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures, parent,
                 locations, action):
        self.id: int = id
        super(VRPState, self).__init__(
            self, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures
        )

        self.parent: Type[VRPState] = parent
        self.locations: Dict[int, npt.NDArray[Tuple[np.float64]]] = locations
        self.action: Any = action


class CVRP(VRP):
    def __init__(self, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures, capacity):
        super(CVRP, self).__init__(vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures)
        self.capacity: Dict[int, npt.NDArray[np.float64]] = capacity

    def __repr__(self):
        return f"Capacitied vehicle routing problem with {self.vertex_num} vertices, {self.depot_num} depots," + \
                f" {self.edge_num} edges, and {self.vehicle_num} vehicles.\nVertices are {self.vertices};\n" + \
                f"Depots are {self.depots};\nDemand are {self.demand};\nEdges are {self.edges};\nDepartures are" + \
                f" {self.departures};\nCapacity are {self.capacity}.\n"

class CVRPState(VRPState):
    def __init__(self, id, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures, capacity,
                 parent, locations, action, load):
        super(VRPState, self).__init__(
            self, id, vertex_num, depot_num, edge_num, vehicle_num, vertices, demand, edges, departures, parent,
            locations, action
        )
        self.capacity: Dict[int, npt.NDArray[np.float64]] = capacity
        self.load: Dict[int, npt.NDArray[np.float64]] = load
