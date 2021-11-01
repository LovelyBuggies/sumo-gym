import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, Any


VEHICLE_XML_TAG = 'trip'

VERTEX_XML_TAG = 'junction'
VERTEX_XML_INVALID_TYPE = 'internal'

EDGE_XML_TAG = 'edge'
EDGE_XML_PRIORITY = '-1'

def encode_xml(file_path):
    pass

def decode_xml(net_xml_file_path: str = None, demand_xml_file_path: str = None):
    """
    Parse the net.xml and rou.xml generated from SUMO and read into VRP initialization environments.
    Return objects: vertices, edges, departures for VRP class
    """

    net_xml_source = open(net_xml_file_path) 
    demand_xml_source = open(demand_xml_file_path) 

    vertices, edge_id_map, edges = _parse_network_xml(net_xml_source)
    departures = _parse_demand_xml(demand_xml_source, edge_id_map, edges)

    net_xml_source.close()
    demand_xml_source.close()

    return np.array(vertices), np.array(edges), np.array(departures)

def _parse_demand_xml(demand_file_path: str, edge_id_mapDict: Dict[str, int], edges: Any):
    """
    :param demand_file_path:      file path of rou.xml
    :param edge_id_map:           sample structure: {'genE0': 0, 'genE1': 1}
    :param edges:                 tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    """
    demand_tree = ET.parse(demand_file_path)
    demand_xml_root = demand_tree.getroot()

    departures = []   # np int array
    for vehicle_trip in demand_xml_root.findall(VEHICLE_XML_TAG):
        departure_edge = vehicle_trip.get('from')
        departures.append(edges[edge_id_map[departure_edge]][0])    

    return departures

def _parse_network_xml(network_file_path: str):
    """
    :param network_file_path:     file path of net.xml
    """
    network_tree = ET.parse(network_file_path)
    network_xml_data = network_tree.getroot()

    vertices_id_map = {} # sample structure: {'genJ1': 0, 'genJ10': 1}
    vertices = [] # tuple of x,y position of each vertex
    vertex_count = 0
    for junction in network_xml_data.findall(VERTEX_XML_TAG):
        if (junction.get('type') != VERTEX_XML_INVALID_TYPE):
            vertices.append([float(junction.get('x')), float(junction.get('y'))])
            vertices_id_map[junction.get('id')] = vertex_count
            vertex_count += 1

    edge_id_map = {} # sample structure: {'genE0': 0, 'genE1': 1}
    edges = [] # tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    edge_count = 0
    for edge in network_xml_data.findall(EDGE_XML_TAG):
        if (edge.get('priority') == EDGE_XML_PRIORITY):
            edges.append([vertices_id_map[edge.get('from')], vertices_id_map[edge.get('to')]])
            edge_id_map[edge.get('id')] = edge_count
            edge_count += 1

    return vertices, edge_id_map, edges