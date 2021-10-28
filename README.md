# sumo-gym

OpenAI-gym like toolkit for developing and comparing reinforcement learning algorithms on SUMO.

P.S. *Private until being a wheel (probably never)*.


## Installation

This software is under active development, it has not been published on PyPI, and some functions are still unstable. If you want to test and contribute to it, you can try this:

```shell
$ python3 -m venv env
$ source env/bin/activate
(venv)$ pip install -e .
(venv)$ python -m ipykernel install --user --name sumo_gym
(venv)$ pytest tests/*.py
```

## Features

SUMO-gym aims to build an interface between SUMO and Reinforcement Learning. With this toolkit, you will be able to convert the data generated from SUMO simulator into RL training setting like OpenAI-gym. Remarkable features include:

1. Auto-generate VRP data from SUMO: `utils/net_xml_decoder.py`;
2. OpenAI-gym like RL training environment;
3. TBD (novel RL algorithm).

