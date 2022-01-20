import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, Any, List, Tuple
import numpy.typing as npt
import sys

VEHICLE_XML_TAG = "vehicle"
VEHICLE_CAPACITY_TAG = "personNumber"

VERTEX_XML_TAG = "junction"
VERTEX_XML_INVALID_TYPE = "internal"
VERTEX_CUSTOMIZED_PARAM = "param"
VERTEX_DEMAND_KEY = "destination"

EDGE_XML_TAG = "edge"
EDGE_XML_INVALID_FUNC = "internal"
EDGE_XML_PRIORITY = "-1"


def get_electric_vehicles(flow_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns electric vehicles

    Each EV is [id (str), maximum speed (float), maximumBatteryCapacity (float)]
    """

    # there should only be one vType
    # defined in rou.xml

    # in this model, all EV's start off fully charged
    vtype = flow_xml_tree.findall("vType")[0]
    vtype_params = vtype.findall("param")
    maxBatteryCap = -1
    for vtype_param in vtype_params:
        if vtype_param.attrib["key"] == "maximumBatteryCapacity":
            maxBatteryCap = float(vtype_param.attrib["value"])

    ev_lst = []
    vehicles = flow_xml_tree.findall(VEHICLE_XML_TAG)
    for vehicle in vehicles:
        ev_lst.append(
            [vehicle.attrib["id"], float(vtype.attrib["maxSpeed"]), maxBatteryCap]
        )
    return ev_lst


def get_charging_stations(additional_xml_tree, net_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns charging stations

    Each charging station is [id, (x_coord, y_coord), edge_id, charging speed]
    """
    cs_lst = []
    stations = additional_xml_tree.findall("chargingStation")
    for station in stations:

        if "shadow" not in station.attrib["id"]:

            # get approximate location
            x_coord, y_coord = station.findall("param")[0].attrib["value"].split()
            # get edge_id
            lane_id = station.attrib["lane"]
            edge_id = None
            # get all edges in net.xml
            edges = net_xml_tree.findall("edge")
            for edge in edges:
                lanes = edge.findall("lane")
                lanes = [
                    lane.attrib["id"] for lane in lanes if lane.attrib["id"] == lane_id
                ]
                if len(lanes) == 1:
                    edge_id = edge.attrib["id"]
                    break
            cs_lst.append(
                (
                    station.attrib["id"],
                    (float(x_coord), float(y_coord)),
                    edge_id,
                    float(station.attrib["power"]),
                    float(station.attrib["endPos"]),
                )
            )

    return cs_lst


def get_vertices(net_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns vertices
    Each vertex is [id (str), x_coord (float), y_coord (float)]
    """
    vtx_lst = []
    junctions = net_xml_tree.findall(VERTEX_XML_TAG)

    for junction in junctions:
        if junction.attrib["type"] == VERTEX_XML_INVALID_TYPE:
            continue
        vtx_lst.append(
            [
                junction.attrib["id"],
                float(junction.attrib["x"]),
                float(junction.attrib["y"]),
            ]
        )

    return vtx_lst


def get_edges(net_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns edges

    Each edge is [id (str), from_vertex_id (str),
                  to_vertex_id (str),
                  edge_length (float)]
    """
    edge_lst = []
    edges = net_xml_tree.findall(EDGE_XML_TAG)
    for e in edges:
        if "function" in e.attrib and e.attrib["function"] == EDGE_XML_INVALID_FUNC:
            continue
        # Edge lengths are given by lane lengths. Each edge has at
        # least one lane and all lanes of an edge have
        # the same length
        lane = e.findall("lane")[0]
        edge_lst.append(
            [
                e.attrib["id"],
                e.attrib["from"],
                e.attrib["to"],
                float(lane.attrib["length"]),
            ]
        )
    return edge_lst


def get_departures(flow_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns departures for each vehicle

    Each departure is [vehicle_id, starting_edge_id]
    and should be defined for all vehicles
    """
    departures = []  # id, start_index tuple
    for vehicle in flow_xml_tree.findall(VEHICLE_XML_TAG):
        route = vehicle.findall("route")[0]  # findall should return one
        edges = route.attrib["edges"]  # space separated list of edge ids
        start_edge = edges.split()[0]
        departures.append([vehicle.attrib["id"], start_edge])

    return departures


def get_demand(net_xml_tree):
    """
    Some junctions are customer nodes and this is represented by
    a junction having param destination
    Each demand is [junction_id, dest_vertex_id]
    """
    demand = []
    for junction in net_xml_tree.findall(VERTEX_XML_TAG):
        if junction.get("type") != VERTEX_XML_INVALID_TYPE:
            for customized_params in junction.findall(VERTEX_CUSTOMIZED_PARAM):
                if customized_params.get("key") == VERTEX_DEMAND_KEY:
                    demand.append([junction.get("id"), customized_params.get("value")])

    return demand


def decode_xml_fmp(
    net_xml_file_path: str = None,
    flow_xml_file_path: str = None,
    additional_xml_path: str = None,
):
    """
    Parse files generated from SUMO and return a FMP instance

    Returns vertices, charging_stations, electric_vehicles,
    edges, departures, and demands
    """
    net_xml_tree = ET.parse(net_xml_file_path)
    flow_xml_tree = ET.parse(flow_xml_file_path)
    additional_xml_tree = ET.parse(additional_xml_path)
    vertices = get_vertices(net_xml_tree)
    charging_stations = get_charging_stations(additional_xml_tree, net_xml_tree)
    electric_vehicles = get_electric_vehicles(flow_xml_tree)
    edges = get_edges(net_xml_tree)
    departures = get_departures(flow_xml_tree)
    demand = get_demand(net_xml_tree)
    return vertices, charging_stations, electric_vehicles, edges, departures, demand


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
