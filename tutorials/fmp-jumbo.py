import sys
import gym
import sumo_gym

if __name__ == "__main__":
    env = gym.make(
        "FMP-v0",
        mode="sumo_config",
        sumo_config_path="assets/data/jumbo/jumbo.sumocfg",
        net_xml_file_path="assets/data/jumbo/jumbo.net.xml",
        demand_xml_file_path="assets/data/jumbo/jumbo.rou.xml",
        additional_xml_file_path="assets/data/jumbo/jumbo.cs.add.xml",
        render_env=True
        if str(sys.argv[sys.argv.index("--render") + 1]) == "1"
        else False,
    )

    for i_episode in range(1):
        observation = env.reset()
        for t in range(3000):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print("Takes actions: ", observation["Takes_action"])
            if done:
                print("Episode finished after {} timesteps.\n".format(t + 1))
                break

    env.close()
