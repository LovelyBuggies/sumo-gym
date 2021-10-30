from gym.envs.registration import register
from sumo_gym.envs.vrp import VRP, VRPEnv
from sumo_gym.spaces.network import Network
from sumo_gym.utils.xml_utils import encode_xml, decoder_xml
from sumo_gym.utils.network_utils import calculate_dist, get_adj_list
from typing import Tuple


__all__ = (
    "VRP",
    "VRPEnv",
    "Network",
    "encode_xml",
    "decoder_xml",
    "calculate_dist",
    "get_adj_list",
)

def __dir__() -> Tuple[str, ...]:
    return __all__

register(
    id='VRP-v0',
    entry_point='sumo_gym.envs:VRPEnv',
)

register(
    id='CVRP-v0',
    entry_point='sumo_gym.envs:CVRPEnv',
)