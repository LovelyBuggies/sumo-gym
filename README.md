# sumo-gym

OpenAI-gym like toolkit for developing and comparing reinforcement learning algorithms on SUMO.

P.S. *Private until being a wheel (probably never)*.


## Installation

This software is under active development, it has not been published on PyPI, and some functions are still unstable. If you want to test and contribute to it, you can try this:

```shell
$ python3 -m venv env
$ source env/bin/activate
(venv)$ pip install -e .
#(venv)$ python -m ipykernel install --user --name sumo_gym
(venv)$ pytest tests/
```

## Features

![](https://tva1.sinaimg.cn/large/008i3skNgy1gvvnprykh1j30yq0u0ace.jpg)

SUMO-gym aims to build an interface between SUMO and Reinforcement Learning. With this toolkit, you will be able to convert the data generated from SUMO simulator into RL training setting like OpenAI-gym. Remarkable features include:

1. Automatically import/export data from/to SUMO: `utils/net_xml_encoder.py` / `utils/net_xml_decoder.py`;
2. OpenAI-gym like RL training environment:

```python
import gym
import sumo_gym
env = gym.make("VRP-v0")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # todo
  observation, reward, done, info = env.step(action) # todo
  if done:
    observation = env.reset()

env.close()
```



