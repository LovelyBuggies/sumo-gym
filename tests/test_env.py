import gym
import sumo_gym

def test_env_basics():
    env = gym.make('VRP-v0') # gym.make('CVRP-v0')
    env.reset()
    env.render()
    env.close()