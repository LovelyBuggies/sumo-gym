# SUMO-gym

[![Actions Status][actions-badge]][actions-link]
[![pre-commit.ci status][pre-commit-badge]][pre-commit-link]
[![Code style: black][black-badge]][black-link]
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?)](#contributors-)

OpenAI-gym like toolkit for developing and comparing reinforcement learning algorithms on SUMO.

![](./assets/workflow.png)


## Installation

This software is under active development, it has not been published on PyPI, and some functions are still unstable. If you want to test and contribute to it, you can try this:

```shell
$ python3 -m venv env
$ source env/bin/activate
(env)$ pip install -r requirements.txt
(env)$ pip install -e .
(env)$ pytest tests/
#(env)$ python -m ipykernel install --user --name sumo_gym
```

## Features

SUMO-gym aims to build an interface between SUMO and Reinforcement Learning. With this toolkit, you will be able to convert the data generated from SUMO simulator into RL training setting like OpenAI-gym. 

**Remarkable features include:**

1. Interface between SUMO simulator and OpenAI-gym liked RL training environment;

```python
import gym
from sumo_gym.envs.fmp import FMP

env = gym.make("FMP-v0", \
n_vertex, n_edge, n_vehicle, n_electric_vehicles, n_charging_station, \
vertices, demand, edges, \
electric_vehicles, departures, charging_stations)
for _ in range(n_episode):
    obs = env.reset()
    for t in range(n_timestamp):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
env.close()
```

2. Visualization rendering tools based on matplotlib for urban mobility problems.

![](./assets/obs.png)

3. Deep reinforcement learning models for urban mobility problems (WIP).

P.S. *Will be a wheel later*.

## Contributors ✨

<table>
  <tr>
    <td align="center"><a href="https://github.com/LovelyBuggies"><img src="https://avatars.githubusercontent.com/u/29083689?v=4?s=100" width="100px;" alt=""/><br /><sub><b>N!no</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/commits?author=LovelyBuggies" title="Code">💻</a> <a href="https://github.com/LovelyBuggies/sumo-gym/issues?q=author%3ALovelyBuggies" title="Bug reports">🐛</a> <a href="#ideas-LovelyBuggies" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/yunhao-wang-871364aa/"><img src="https://avatars.githubusercontent.com/u/18152628?v=4?s=100" width="100px;" alt=""/><br /><sub><b>yunhaow</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/commits?author=wyunhao" title="Code">💻</a> <a href="https://github.com/LovelyBuggies/sumo-gym/issues?q=author%3Awyunhao" title="Bug reports">🐛</a> <a href="#ideas-wyunhao" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/qqqube"><img src="https://avatars.githubusercontent.com/u/24397793?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lauren Hong</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/commits?author=qqqube" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/AlwaysSearching"><img src="https://avatars.githubusercontent.com/u/53829883?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sam Fieldman</b></sub></a><br /><a href="https://github.com/LovelyBuggies/sumo-gym/issues?q=author%3AAlwaysSearching" title="Bug reports">🐛</a></td>
  </tr>
</table>


[actions-badge]:            https://github.com/LovelyBuggies/sumo-gym/workflows/CI/badge.svg
[actions-link]:             https://github.com/LovelyBuggies/sumo-gym/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[pre-commit-badge]:         https://results.pre-commit.ci/badge/github/LovelyBuggies/sumo-gym/main.svg
[pre-commit-link]:          https://results.pre-commit.ci/repo/github/LovelyBuggies/sumo-gym
