import sys
import gym

if __name__ == "__main__":
    env = gym.make(
        "FMP-v0",
        sumo_gui_path=sys.argv[sys.argv.index("--sumo-gui-path") + 1],
        sumo_configuration_path=sys.argv[sys.argv.index("--sumo-config-path") + 1],
        net_xml_file_path="assets/data/jumbo.net.xml",
        demand_xml_file_path="assets/data/jumbo.rou.xml",
        additional_xml_file_path="assets/data/jumbo_charging_station_additional.xml",
    )

    for i_episode in range(1):
        observation = env.reset()
        for t in range(1500):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            env.render()
            if done:
                print("Episode finished after {} timesteps.\n".format(t + 1))
                break

    env.close()
