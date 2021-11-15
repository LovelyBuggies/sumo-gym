from gym.envs.registration import register
from sumo_gym.envs.vrp import VRP, VRPEnv
from sumo_gym.spaces.network import NetworkSpace
from sumo_gym.utils.xml_utils import encode_xml, decode_xml
from sumo_gym.utils.network_utils import calculate_dist, get_adj_list
from sumo_gym.utils.svg_uitls import vehicle_marker
from typing import Tuple


__all__ = (
    "VRP",
    "VRPEnv",
    "NetworkSpace",
    "encode_xml",
    "decode_xml",
    "calculate_dist",
    "get_adj_list",
    "vehicle_marker",
)


def __dir__() -> Tuple[str, ...]:
    return __all__


register(
    id="VRP-v0",
    entry_point="sumo_gym.envs:VRPEnv",
)

register(
    id="FMP-v0",
    entry_point="sumo_gym.envs:FMPEnv",
)
