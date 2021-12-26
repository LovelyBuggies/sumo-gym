# SUMO-Gym Contribution Guidelines

At this time we are currently accepting the current forms of contributions:

- Bug reports (keep in mind that changing environment behavior should be minimized as that requires releasing a new version of the environment and makes results hard to compare across versions)
- Pull requests for bug fixes
- Documentation improvements

The commits are supposed to follow [Conventional Commits.](https://www.conventionalcommits.org/)

# Pre-commit

Refer [.pre-commit-config.yaml](./.pre-commit-config.yaml) to see what are the hooks.

# Develop

We welcome enthusiastic contributors, and you can develop using Python virtual environment:

```shell
$ python3 -m venv .env
$ source .env/bin/activate
(.env)$ pip install --upgrade pip
(.env)$ pip install -r requirements.txt
(.env)$ pip install -e .
#(.env)$ pytest tests/
#(.env)$ python -m ipykernel install --user --name sumo_gym
(.env)$ touch ~/.bashrc; open ~/.bashrc
(.env)$ export SUMO_HOME=<your_path_to>/sumo SUMO_GUI_PATH=<your_path_to>/sumo-gui # and copy the paths to ~/.bashrc
(.env)$ python3 tutorials/fmp-jumbo.py --render 0
```

You can also use Anaconda virtual environment to develop:

```shell
todo
```