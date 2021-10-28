import gym
import sumo_gym
env = gym.make('VRP-v0')
observation = env.reset()
# for _ in range(10):
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     if done:
#     observation = env.reset()

env.render()
env.close()
