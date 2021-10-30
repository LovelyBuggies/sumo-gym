from gym.envs.registration import register
import sumo_gym.envs
import sumo_gym.utils

register(
    id='VRP-v0',
    entry_point='sumo_gym.envs:VRPEnv',
)

register(
    id='CVRP-v0',
    entry_point='sumo_gym.envs:CVRPEnv',
)