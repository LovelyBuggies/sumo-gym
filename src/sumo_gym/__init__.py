from gym.envs.registration import register


register(
    id='VRP-v0',
    entry_point='sumo_gym.envs:VRPEnv',
)


register(
    id='CVRP-v0',
    entry_point='sumo_gym.envs:CVRPEnv',
)