"""
Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates
You can visualize experiment results in ~/ray_results using TensorBoard.
Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import gym
from gym.spaces import Discrete, Box, Dict
import numpy as np
import os
import random
from sumo_gym.utils.fmp_utils import Vertex, Edge, Demand, ChargingStation, ElectricVehicles
from sumo_gym.envs.fmp import FMPEnv

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

from matplotlib import pyplot as plt

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=50,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=0.60,
    help="Reward at which we stop training.")
parser.add_argument(
    "--no-tune",
    action="store_false",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")


# class SimpleCorridor(gym.Env):
#     """Example of a custom env in which you have to walk down a corridor.
#     You can configure the length of the corridor via the env config."""
#
#     def __init__(self, config: EnvContext):
#         self.end_pos = config["corridor_length"]
#         self.cur_pos = 0
#         self.action_space = Discrete(2)
#         #self.observation_space = Dict({"observation_self":Box(
#         #    0.0, self.end_pos, shape=(1, ), dtype=np.float32)})
#         self.observation_space = Box(
#             0.0, self.end_pos, shape=(1, ), dtype=np.float32)
#         # Set the seed. This is only used for the final (reach goal) reward.
#         self.seed(config.worker_index * config.num_workers)
#
#     def reset(self):
#         self.cur_pos = 0
#         return [self.cur_pos]
#
#     def step(self, action):
#         assert action in [0, 1], action
#         if action == 0 and self.cur_pos > 0:
#             self.cur_pos -= 1
#         elif action == 1:
#             self.cur_pos += 1
#         done = self.cur_pos >= self.end_pos
#         # Produce a random reward when we reach the goal.
#         return [self.cur_pos], \
#             random.random() * 2 if done else -0.1, done, {}
#
#     def seed(self, seed=None):
#         random.seed(seed)

vertices = np.asarray(
    [
        Vertex(0.0, 0.0),
        Vertex(0.0, 1.0),
        Vertex(0.0, 2.0),
        Vertex(0.0, 3.0),
        Vertex(0.0, 4.0),
        Vertex(0.0, 5.0),
        Vertex(1.0, 0.0),
        Vertex(1.0, 1.0),
        Vertex(1.0, 2.0),
        Vertex(1.0, 3.0),
        Vertex(1.0, 4.0),
        Vertex(1.0, 5.0),
        Vertex(2.0, 0.0),
        Vertex(2.0, 1.0),
        Vertex(2.0, 2.0),
        Vertex(2.0, 3.0),
        Vertex(2.0, 4.0),
        Vertex(2.0, 5.0),
        Vertex(3.0, 0.0),
        Vertex(3.0, 1.0),
        Vertex(3.0, 2.0),
        Vertex(3.0, 3.0),
        Vertex(3.0, 4.0),
        Vertex(3.0, 5.0),
        Vertex(4.0, 0.0),
        Vertex(4.0, 1.0),
        Vertex(4.0, 2.0),
        Vertex(4.0, 3.0),
        Vertex(4.0, 4.0),
        Vertex(4.0, 5.0),
        Vertex(5.0, 0.0),
        Vertex(5.0, 1.0),
        Vertex(5.0, 2.0),
        Vertex(5.0, 3.0),
        Vertex(5.0, 4.0),
        Vertex(5.0, 5.0),
    ]
)

edges = np.asarray(
    [
        Edge(0, 1),
        Edge(1, 0),
        Edge(1, 2),
        Edge(2, 1),
        Edge(2, 3),
        Edge(3, 2),
        Edge(3, 4),
        Edge(4, 3),
        Edge(4, 5),
        Edge(5, 4),
        Edge(0, 6),
        Edge(6, 0),
        Edge(1, 7),
        Edge(7, 1),
        Edge(2, 8),
        Edge(8, 2),
        Edge(3, 9),
        Edge(9, 3),
        Edge(4, 10),
        Edge(10, 4),
        Edge(5, 11),
        Edge(11, 5),
        Edge(6, 7),
        Edge(7, 6),
        Edge(7, 8),
        Edge(8, 7),
        Edge(8, 9),
        Edge(9, 8),
        Edge(9, 10),
        Edge(10, 9),
        Edge(10, 11),
        Edge(11, 10),
        Edge(6, 12),
        Edge(12, 6),
        Edge(7, 13),
        Edge(13, 7),
        Edge(8, 14),
        Edge(14, 8),
        Edge(9, 15),
        Edge(15, 9),
        Edge(10, 16),
        Edge(16, 10),
        Edge(11, 17),
        Edge(17, 11),
        Edge(12, 13),
        Edge(13, 12),
        Edge(13, 14),
        Edge(14, 13),
        Edge(14, 15),
        Edge(15, 14),
        Edge(15, 16),
        Edge(16, 15),
        Edge(16, 17),
        Edge(17, 16),
        Edge(12, 18),
        Edge(18, 12),
        Edge(13, 19),
        Edge(19, 13),
        Edge(14, 20),
        Edge(20, 14),
        Edge(15, 21),
        Edge(21, 15),
        Edge(16, 22),
        Edge(22, 16),
        Edge(17, 23),
        Edge(23, 17),
        Edge(18, 19),
        Edge(19, 18),
        Edge(19, 20),
        Edge(20, 19),
        Edge(20, 21),
        Edge(21, 20),
        Edge(21, 22),
        Edge(22, 21),
        Edge(22, 23),
        Edge(23, 22),
        Edge(18, 24),
        Edge(24, 18),
        Edge(19, 25),
        Edge(25, 19),
        Edge(20, 26),
        Edge(26, 20),
        Edge(21, 27),
        Edge(27, 21),
        Edge(22, 28),
        Edge(28, 22),
        Edge(23, 29),
        Edge(29, 23),
        Edge(24, 25),
        Edge(25, 24),
        Edge(25, 26),
        Edge(26, 25),
        Edge(26, 27),
        Edge(27, 26),
        Edge(27, 28),
        Edge(28, 27),
        Edge(28, 29),
        Edge(29, 28),
        Edge(24, 30),
        Edge(30, 24),
        Edge(25, 31),
        Edge(31, 25),
        Edge(26, 32),
        Edge(32, 26),
        Edge(27, 33),
        Edge(33, 27),
        Edge(28, 34),
        Edge(34, 28),
        Edge(29, 35),
        Edge(35, 29),
        Edge(30, 31),
        Edge(31, 30),
        Edge(31, 32),
        Edge(32, 31),
        Edge(32, 33),
        Edge(33, 32),
        Edge(33, 34),
        Edge(34, 33),
        Edge(34, 35),
        Edge(35, 34),
    ]
)

n_vertex = len(vertices)
n_edge = len(edges)
n_vehicle = 1
n_electric_vehicle = 1
n_charging_station = 3
electric_vehicles = np.asarray(
    [ElectricVehicles(i, 1, 220, 50) for i in range(n_electric_vehicle)]
)
charging_stations = np.asarray(
    [
        ChargingStation(3, 220, 20),
        ChargingStation(33, 220, 20),
        ChargingStation(22, 220, 30),
    ]
)
available_vertices = np.asarray(
    [
        v
        for v in range(35)
        if v
        not in (charging_station.location for charging_station in charging_stations)
    ]
)
departures = np.asarray([19])
demand = np.asarray(
    [
        Demand(6, 4),
        Demand(5, 16),
        Demand(13, 20),
        Demand(28, 11),
        Demand(12, 30),
        Demand(27, 5),
        Demand(1, 28),
        Demand(13, 24),
        Demand(19, 18),
        Demand(11, 32),
    ]
)


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


def plot_rewards(max_rewards, mean_rewards, min_rewards):
    plt.plot(max_rewards, label="max_rewards")
    plt.plot(mean_rewards, label="mean_rewards")
    plt.plot(min_rewards, label="min_rewards")
    plt.legend()
    plt.savefig('reward.png')

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel
        if args.framework == "torch" else CustomModel)

    config = {
        "env": FMPEnv,  # or "corridor" if registered above
        "env_config": {
            "mode": "numerical",
            "n_vertex": n_vertex,
            "n_edge": n_edge,
            "n_vehicle": n_vehicle,
            "n_electric_vehicle": n_electric_vehicle,
            "n_charging_station": n_charging_station,
            "vertices": vertices,
            "demand": demand,
            "edges": edges,
            "electric_vehicles": electric_vehicles,
            "departures": departures,
            "charging_stations": charging_stations,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "num_workers": 1,  # parallelism
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    max_rewards = []
    mean_rewards = []
    min_rewards = []
    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        # use fixed learning rate instead of grid search (needs tune)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPOTrainer(config=ppo_config, env=FMPEnv)
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            max_rewards.append(result["episode_reward_max"])
            mean_rewards.append(result["episode_reward_mean"])
            min_rewards.append(result["episode_reward_min"])

            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps or \
                    result["episode_reward_mean"] >= args.stop_reward:
                break
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run(args.run, config=config, stop=stop)
        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    plot_rewards(max_rewards, mean_rewards, min_rewards)
    ray.shutdown()
