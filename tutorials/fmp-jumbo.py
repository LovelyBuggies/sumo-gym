import sys
import gym
import sumo_gym

if __name__ == "__main__":
    env = gym.make(
        "FMP-v0",
        # sumo_configuration_path=sys.argv[sys.argv.index("--sumo-config-path") + 1],
        net_xml_file_path="../assets/data/jumbo/jumbo.net.xml",
        demand_xml_file_path="../assets/data/jumbo/jumbo.rou.xml",
        additional_xml_file_path="../assets/data/jumbo/jumbo.cs.add.xml",
    )

    for i_episode in range(1):
        observation = env.reset()
        for t in range(3000):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(observation["Takes_action"])
            # env.render()
            if done:
                print("Episode finished after {} timesteps.\n".format(t + 1))
                break

    env.close()
