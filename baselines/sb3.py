from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

import gym
import numpy as np
from sumo_gym.utils.fmp_utils import (
    Vertex,
    Edge,
    Demand,
    ChargingStation,
    ElectricVehicles,
)
from sumo_gym.envs.fmp import FMPEnv


# run sumo network without rendering
env = gym.make(
    "FMP-v0",
    mode="sumo_config",
    sumo_config_path="assets/data/jumbo/jumbo.sumocfg",
    net_xml_file_path="assets/data/jumbo/jumbo.net.xml",
    demand_xml_file_path="assets/data/jumbo/jumbo.rou.xml",
    additional_xml_file_path="assets/data/jumbo/jumbo.cs.add.xml",
    render_env=False,
)


check_env(env, warn=True)

env = make_vec_env(lambda: env, n_envs=1)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./assets/tensorboards/fmpenv/jumbo/ppo/",
)
# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./assets/tensorboards/fmpenv/a2c/")
# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./assets/tensorboards/fmpenv/dqn/")
model.learn(total_timesteps=10000, tb_log_name="mlp_policy", reset_num_timesteps=False)

model.save("./assets/models/fmpenv/jumbo/ppo")

obs = env.reset()
n_steps = 4
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    if done:
        print("Goal reached!", "reward=", reward)
        break
