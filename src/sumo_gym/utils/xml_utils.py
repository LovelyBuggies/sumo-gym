import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, Any, List, Tuple
import numpy.typing as npt
import sys

VEHICLE_XML_TAG = "trip"
VEHICLE_CAPACITY_TAG = "personNumber"

VERTEX_XML_TAG = "junction"
VERTEX_XML_INVALID_TYPE = "internal"
VERTEX_CUSTOMIZED_PARAM = "param"
VERTEX_DEMAND_KEY = "destination"

EDGE_XML_TAG = "edge"
EDGE_XML_PRIORITY = "-1"


def encode_xml_fmp(net_xml_file_path: str = None, flow_xml_file_path: str = None):
    # TODO
    pass


def get_electric_vehicles(battery_xml_tree, flow_xml_tree, init_iter):
    """
    Helper function for decode_xml_fmp

    https://sumo.dlr.de/docs/Models/Electric.html#battery-output

    Returns electric vehicles
    """
    if not init_iter:
        ev_lst = []
        timesteps = battery_xml_tree.findall("timestep")
        latest_timestep = sorted(timesteps, key=lambda ts: ts.attrib["time"])[-1]
        vehicles = latest_timestep.findall("vehicle")
        for vehicle in vehicles:
            atb_ = vehicle.attrib
            ev_lst.append(
                (
                    atb_["id"],
                    float(atb_["actualBatteryCapacity"])
                    / float(atb_["maximumBatteryCapacity"]),
                )
            )

        return np.array(ev_lst)
    ev_lst = []
    vehicles = flow_xml_tree.findall("vehicle")
    for vehicle in vehicles:
        # all vehicles start off with 100% energy level
        ev_lst.append((vehicle.attrib["id"], 1.0))
    return np.asarray(ev_lst)


def get_charging_stations(charging_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns charging stations
    """
    cs_lst = []
    stations = charging_xml_tree.findall("chargingStation")
    for station in stations:
        cs_lst.append((station.attrib["id"], float(station.attrib["chargeDelay"])))
    return np.asarray(cs_lst)


def get_vertices(net_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns vertices
    """
    vtx_lst = []
    junctions = net_xml_tree.findall(VERTEX_XML_TAG)

    for junction in junctions:
        vtx_lst.append(
            [
                junction.attrib["id"],
                float(junction.attrib["x"]),
                float(junction.attrib["y"]),
            ]
        )

    return np.asarray(vtx_lst)


def get_edges(net_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns edges
    """
    edge_lst = []
    edges = net_xml_tree.findall("edge")
    for e in edges:
        if "function" in e.attrib and e.attrib["function"] == VERTEX_XML_INVALID_TYPE:
            continue

        lane = e.findall("lane")[0]
        edge_lst.append(
            [e.attrib["id"], e.attrib["from"], e.attrib["to"], lane.attrib["length"]]
        )
    return np.asarray(edge_lst)


def get_departures(flow_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns departures for each vehicle

    [vehicle_index, starting_vertex_index]
    """

    departures = []  # id, start_index tuple
    for vehicle_trip in flow_xml_tree.findall(VEHICLE_XML_TAG):
        departure_edge = vehicle_trip.get("from")
        departures.append([vehicle_trip.get("id"), departure_edge])

    return np.asarray(departures)

    # departures_lst = []
    # vehicles = flow_xml_tree.findall("vehicle")
    # routes = flow_xml_tree.findall("route")
    # edges = net_xml_tree.findall("edge")

    # routes = flow_xml_tree.findall("routes")
    # edge_map = {}
    # # edge_id to `from` vertex id
    # for edge in edges:
    #     if "function" in edge.attrib and edge.attrib["function"] == VERTEX_XML_INVALID_TYPE:
    #         continue
    #     edge_map[edge.attrib["id"]] = edge.attrib["from"]
    # # route_id to first edge_id
    # route_map = {}
    # for route in routes:
    #     route_map[route.attrib["id"]] = route.attrib["edges"].split()[0]
    # for ev in vehicles:
    #     if "route" in ev.attrib:
    #         route_id = ev.attrib["route"]
    #         edge_id = route_map[route_id]
    #         departures_lst.append([ev.attrib["id"], edge_map[edge_id]])

    # return np.asarray(departures)


def get_demand(net_xml_tree: str = None):
    demand = []  # start, destination id tuple
    for junction in net_xml_tree.findall(VERTEX_XML_TAG):
        if junction.get("type") != VERTEX_XML_INVALID_TYPE:
            for customized_params in junction.findall(VERTEX_CUSTOMIZED_PARAM):
                if customized_params.get("key") == VERTEX_DEMAND_KEY:
                    demand.append([junction.get("id"), customized_params.get("value")])

    return np.asarray(demand)


def decode_xml_fmp(
    net_xml_file_path: str = None,
    flow_xml_file_path: str = None,
    # charging_xml_path: str = None,
    # battery_xml_file_path: str = None,
    init_iter: bool = True,
):
    """
    Parse net.xml, rou.xml, and battery.out.xml files
    generated from SUMO and return a FMP instance
    (if this is the first invocation, no battery.out.xml
    file will be passed in)
    Returns vertices, charging_stations, electric_vehicles,
    edges, and departures
    """
    net_xml_tree = ET.parse(net_xml_file_path)
    flow_xml_tree = ET.parse(flow_xml_file_path)
    # battery_xml_tree = None
    # if battery_xml_file_path:
    #     battery_xml_tree = ET.parse(battery_xml_file_path)
    # charging_xml_tree = ET.parse(charging_xml_path)

    vertices = get_vertices(net_xml_tree)

    # charging_stations = get_charging_stations(charging_xml_tree)

    # electric_vehicles = get_electric_vehicles(battery_xml_tree, flow_xml_tree, init_iter)

    edges = get_edges(net_xml_tree)

    # departures = get_departures(net_xml_tree, flow_xml_tree, init_iter)
    departures = get_departures(flow_xml_tree)

    demand = get_demand(net_xml_tree)

    # return vertices, charging_stations, electric_vehicles, edges, departures
    return vertices, edges, departures, demand


def encode_xml(file_path):
    pass


def decode_xml(
    net_xml_file_path: str = None, flow_xml_file_path: str = None
) -> Tuple[npt.NDArray[Any]]:
    """
    Parse the net.xml and rou.xml generated from SUMO and read into VRP initialization environments.
    Return objects: vertices, demand, edges, departures, capacity for VRP class
    """

    net_xml_source = open(net_xml_file_path)
    flow_xml_source = open(flow_xml_file_path)

    vertices, demand, edge_id_map, edges = _parse_network_xml(net_xml_source)
    departures, capacity = _parse_flow_xml(flow_xml_source, edge_id_map, edges)

    net_xml_source.close()
    flow_xml_source.close()

    return (
        np.asarray(vertices),
        np.asarray(demand),
        np.asarray(edges),
        np.asarray(departures),
        np.asarray(capacity),
    )


def _parse_flow_xml(flow_file_path: str, edge_id_map: Dict[str, int], edges: Any):
    """
    :param flow_file_path:      file path of rou.xml
    :param edge_id_map:         sample structure: {'genE0': 0, 'genE1': 1}
    :param edges:               tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    """
    flow_tree = ET.parse(flow_file_path)
    flow_xml_root = flow_tree.getroot()

    departures = []  # int array
    capacity = []  # float array
    for vehicle_trip in flow_xml_root.findall(VEHICLE_XML_TAG):
        departure_edge = vehicle_trip.get("from")
        departures.append(edges[edge_id_map[departure_edge]][0])

        capacity_value = vehicle_trip.get(VEHICLE_CAPACITY_TAG) or 20.0
        capacity.append(float(capacity_value))

    return departures, capacity


def _parse_network_xml(network_file_path: str):
    """
    :param network_file_path:     file path of net.xml
    """
    network_tree = ET.parse(network_file_path)
    network_xml_data = network_tree.getroot()

    vertices_id_map = {}  # sample structure: {'genJ1': 0, 'genJ10': 1}
    vertices = []  # tuple of x,y position of each vertex
    demand = []  # float array
    vertex_count = 0
    for junction in network_xml_data.findall(VERTEX_XML_TAG):
        if junction.get("type") != VERTEX_XML_INVALID_TYPE:
            vertices.append([float(junction.get("x")), float(junction.get("y"))])
            vertices_id_map[junction.get("id")] = vertex_count

            demand_value = 0.0
            for customized_params in junction.findall(VERTEX_CUSTOMIZED_PARAM):
                if customized_params.get("key") == VERTEX_DEMAND_KEY:
                    demand_value = float(customized_params.get("value"))
            demand.append(demand_value)

            vertex_count += 1

    edge_id_map = {}  # sample structure: {'genE0': 0, 'genE1': 1}
    edges = (
        []
    )  # tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    edge_count = 0
    for edge in network_xml_data.findall(EDGE_XML_TAG):
        if edge.get("priority") == EDGE_XML_PRIORITY:
            edges.append(
                [vertices_id_map[edge.get("from")], vertices_id_map[edge.get("to")]]
            )
            edge_id_map[edge.get("id")] = edge_count
            edge_count += 1

    return vertices, demand, edge_id_map, edges
