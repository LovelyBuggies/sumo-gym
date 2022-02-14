import gym
import sumo_gym
import sys
from pettingzoo.test import api_test
from pettingzoo.utils import wrappers


env = gym.make(
    "FMP-v0",
    mode="sumo_config",
    verbose=1,
    sumo_config_path="assets/data/jumbo/jumbo.sumocfg",
    net_xml_file_path="assets/data/jumbo/jumbo.net.xml",
    demand_xml_file_path="assets/data/jumbo/jumbo.rou.xml",
    additional_xml_file_path="assets/data/jumbo/jumbo.cs.add.xml",
    render_env=True
        if str(sys.argv[sys.argv.index("--render") + 1]) == "1"
        else False,
)

# api_test(env, num_cycles=10, verbose_progress=False)

for _ in range(1):
    env.reset()
    # will auto-complete and thus break the loop when self.dones are all True
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = env.action_space(agent).sample()
        env.step(action)
        env.render()

env.close()
