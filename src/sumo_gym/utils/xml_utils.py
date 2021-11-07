import sys
import xml.etree.ElementTree as ET
import numpy as np
#from sumo-gym.src.sumo_gym.envs.fmp import FMP
from typing import Dict, Any, List, Tuple
import numpy.typing as npt

VEHICLE_XML_TAG = 'trip'
VEHICLE_CAPACITY_TAG = 'personNumber'

VERTEX_XML_TAG = 'junction'
VERTEX_XML_INVALID_TYPE = 'internal'
VERTEX_CUSTOMIZED_PARAM = 'param'
VERTEX_DEMAND_KEY = 'demand'

EDGE_XML_TAG = 'edge'
EDGE_XML_PRIORITY = '-1'

def encode_xml_fmp(net_xml_file_path: str = None, flow_xml_file_path: str = None):
    # TODO
    pass



def get_edges_and_lanes(net_xml_tree):
    """
    Given next_xml_tree, 
    return dictionary of edges and lanes

    Assumes no edges contain stopOffset elements
    """
    # map edge id to dictionary with attributes
    edges_and_lanes = {}

    edge_attr = ["from", "to", "priority", "function"]
    lane_attr = ["index", "speed", "length", "shape"]

    edges = net_xml_tree.findall("edge")
    print(edges)
    for edge in edges:

        edge_id = edge.attrib["id"]
        val = edge.attrib
        del val["id"]
        
        if "priority" in val:
            val["priority"] = int(val["priority"])
        
        edges_and_lanes[edge_id] = val

        lanes = edge.findall("lane")

        for lane in lanes:

            val = lane.attrib
            
            if "index" in val:
                val["index"] = int(val["index"])
            
            for float_attr in ["speed", "length"]:
                if float_attr in val:
                    val[float_attr] = float(val[float_attr])

            lane_list = edges_and_lanes[edge_id].get("lanes", [])
            lane_list.append(val)
            edges_and_lanes[edge_id]["lanes"] = lane_list

    return edges_and_lanes



def decode_xml_fmp(net_xml_file_path: str = None):
    """
    Parse net.xml file from SUMO and create FMP instance

    https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html
    """
    net_xml_tree = ET.parse(net_xml_file_path)
    
    edges_and_lanes = get_edges_and_lanes(net_xml_tree)

    print(edges_and_lanes)




def encode_xml(file_path):
    pass
"""
def decode_xml(net_xml_file_path: str = None, flow_xml_file_path: str = None) -> Tuple[npt.NDArray[Any]]:
    
    Parse the net.xml and rou.xml generated from SUMO and read into VRP initialization environments.
    Return objects: vertices, demand, edges, departures, capacity for VRP class
    

    net_xml_source = open(net_xml_file_path) 
    flow_xml_source = open(flow_xml_file_path) 

    vertices, demand, edge_id_map, edges = _parse_network_xml(net_xml_source)
    departures, capacity = _parse_flow_xml(flow_xml_source, edge_id_map, edges)

    net_xml_source.close()
    flow_xml_source.close()

    return np.asarray(vertices), np.asarray(demand), np.asarray(edges), np.asarray(departures), np.asarray(capacity)
"""
def _parse_flow_xml(flow_file_path: str, edge_id_map: Dict[str, int], edges: Any):
    """
    :param flow_file_path:      file path of rou.xml
    :param edge_id_map:         sample structure: {'genE0': 0, 'genE1': 1}
    :param edges:               tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    """
    flow_tree = ET.parse(flow_file_path)
    flow_xml_root = flow_tree.getroot()

    departures = []   # int array
    capacity = [] # float array
    for vehicle_trip in flow_xml_root.findall(VEHICLE_XML_TAG):
        departure_edge = vehicle_trip.get('from')
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

    vertices_id_map = {} # sample structure: {'genJ1': 0, 'genJ10': 1}
    vertices = [] # tuple of x,y position of each vertex
    demand = [] # float array
    vertex_count = 0
    for junction in network_xml_data.findall(VERTEX_XML_TAG):
        if (junction.get('type') != VERTEX_XML_INVALID_TYPE):
            vertices.append([float(junction.get('x')), float(junction.get('y'))])
            vertices_id_map[junction.get('id')] = vertex_count

            demand_value = 0.0
            for customized_params in junction.findall(VERTEX_CUSTOMIZED_PARAM):
                if (customized_params.get('key') == VERTEX_DEMAND_KEY):
                    demand_value = float(customized_params.get('value'))
            demand.append(demand_value)

            vertex_count += 1

    edge_id_map = {} # sample structure: {'genE0': 0, 'genE1': 1}
    edges = [] # tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    edge_count = 0
    for edge in network_xml_data.findall(EDGE_XML_TAG):
        if (edge.get('priority') == EDGE_XML_PRIORITY):
            edges.append([vertices_id_map[edge.get('from')], vertices_id_map[edge.get('to')]])
            edge_id_map[edge.get('id')] = edge_count
            edge_count += 1

    return vertices, demand, edge_id_map, edges


def cli_main():
    """ CLI commands for testing """

    if len(sys.argv) == 1:
        return

    # python3 xml_utils.py decode_xml_fmp {net_xml_file_path}
    if sys.argv[1] == "decode_xml_fmp":
        decode_xml_fmp(sys.argv[2])



# entry point to CLI
if __name__ == "__main__":

    cli_main()











