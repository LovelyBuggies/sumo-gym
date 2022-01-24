# SUMO-gym

[![Actions Status][actions-badge]][actions-link]
[![pre-commit.ci status][pre-commit-badge]][pre-commit-link]
[![Code style: black][black-badge]][black-link]
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?)](#contributors-)

OpenAI-gym like toolkit for developing and comparing reinforcement learning algorithms on SUMO.



## Installation

Install SUMO, SUMO GUI and XQuartz according to [official guide](https://sumo.dlr.de/docs/Installing/index.html#macos).

```shell
$ python3 -m venv .env
$ source .env/bin/activate
(.env)$ pip install -r requirements.txt
(.env)$ pip install sumo-gym
(.env)$ export SUMO_HOME=<your_path_to>/sumo SUMO_GUI_PATH=<your_path_to>/sumo-gui # and copy the paths to ~/.bashrc
```

The installation is successful so far, then you can try the examples in the tutorials, for example:

```shell
(.env)$ python3 tutorials/fmp-jumbo.py --render 0
```

## Features

SUMO-gym aims to build an interface between SUMO and Reinforcement Learning. With this toolkit, you will be able to convert the data generated from SUMO simulator into RL training setting like OpenAI-gym. 

**Remarkable features include:**

1. OpenAI-gym RL training environment based on SUMO.

```python
import gym
from sumo_gym.envs.fmp import FMP

env = gym.make(
    "FMP-v0", mode, n_vertex, n_edge, n_vehicle, 
    n_electric_vehicles, n_charging_station, 
    vertices, demand, edges, 
    electric_vehicles, departures, charging_stations,
)
for _ in range(n_episode):
    obs = env.reset()
    for t in range(n_timestamp):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
env.close()
```

2. Rendering tools based on matplotlib for urban mobility problems.

<img src="https://github.com/LovelyBuggies/sumo-gym/blob/main/assets/imgs/sumo-demo.gif?raw=true" width="400"/>

3. Visualization tools that plot the statistics for each observation. 

## Contributors

We would like to acknowledge the contributors that made this project possible ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LovelyBuggies"><img src="https://avatars.githubusercontent.com/u/29083689?v=4?s=80" width="80px;" alt=""/><br /><sub><b>N!no</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/commits?author=LovelyBuggies" title="Code">💻</a> <a href="https://github.com/LovelyBuggies/sumo-gym/issues?q=author%3ALovelyBuggies" title="Bug reports">🐛</a> <a href="#ideas-LovelyBuggies" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/yunhao-wang-871364aa/"><img src="https://avatars.githubusercontent.com/u/18152628?v=4?s=80" width="80px;" alt=""/><br /><sub><b>yunhaow</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/commits?author=wyunhao" title="Code">💻</a> <a href="https://github.com/LovelyBuggies/sumo-gym/issues?q=author%3Awyunhao" title="Bug reports">🐛</a> <a href="#ideas-wyunhao" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/AlwaysSearching"><img src="https://avatars.githubusercontent.com/u/53829883?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Sam Fieldman</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/issues?q=author%3AAlwaysSearching" title="Bug reports">🐛</a> <a href="#ideas-AlwaysSearching" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/qqqube"><img src="https://avatars.githubusercontent.com/u/24397793?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Lauren Hong</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/commits?author=qqqube" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nmauskar"><img src="https://avatars.githubusercontent.com/u/6404257?v=4?s=80" width="80px;" alt=""/><br /><sub><b>nmauskar</b></sub></a><br /> <a href="https://github.com/LovelyBuggies/sumo-gym/commits?author=nmauskar" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.


[actions-badge]:            https://github.com/LovelyBuggies/sumo-gym/workflows/CI/badge.svg
[actions-link]:             https://github.com/LovelyBuggies/sumo-gym/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[pre-commit-badge]:         https://results.pre-commit.ci/badge/github/LovelyBuggies/sumo-gym/main.svg
[pre-commit-link]:          https://results.pre-commit.ci/repo/github/LovelyBuggies/sumo-gym
