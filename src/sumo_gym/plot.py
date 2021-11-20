import sys
import sumo_gym.envs as envs
import numpy as np
from typing import Dict, Any, Union, Set

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
except ModuleNotFoundError:
    print(
        "sumo_gym requires matplotlib to plot, either install sumo_gym[plot] or matplotlib",
        file=sys.stderr,
    )
    raise


def _expand_shortcuts(key: str) -> str:
    if key == "ls":
        return "linestyle"
    return key


def _filter_dict(
    __dict: Dict[str, Any], prefix: str, *, ignore: Union[Set[str], None] = None
) -> Dict[str, Any]:
    """
    Keyword argument conversion: convert the kwargs to several independent args, pulling
    them out of the dict given. Prioritize prefix_kw dict.
    """

    # If passed explicitly, use that
    if f"{prefix}kw" in __dict:
        res: dict[str, Any] = __dict.pop(f"{prefix}kw")
        return {_expand_shortcuts(k): v for k, v in res.items()}

    ignore_set: set[str] = ignore or set()
    return {
        _expand_shortcuts(key[len(prefix) :]): __dict.pop(key)
        for key in list(__dict)
        if key.startswith(prefix) and key not in ignore_set
    }


def plot_VRP(self: envs.vrp.VRP, *, ax: matplotlib.axes.Axes, **kwargs) -> Any:
    x, y = list(), list()
    for v in self.vertices:
        x.append(v[0])
        y.append(v[1])

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(1, 0, 0)

    depot_kwargs = _filter_dict(kwargs, "depot_")
    vertex_kwargs = _filter_dict(kwargs, "vertex_")
    edge_kwargs = _filter_dict(kwargs, "edge_")
    if len(kwargs):
        raise ValueError(f"{set(kwargs)} not needed")

    # avoid alpha=0 when no demand
    if max(self.demand) == 0:
        alpha = np.ones(self.demand.shape) * 0.6
    else:
        alpha = self.demand / max(self.demand) * 0.4 + np.ones(self.demand.shape) * 0.6

    for e in self.edges:
        l1, l2 = self.vertices[e[0]], self.vertices[e[1]]
        ax.plot([l1[0], l2[0]], [l1[1], l2[1]], **edge_kwargs)

    ax.scatter(x[: self.n_depot], y[: self.n_depot], alpha=1, **depot_kwargs)
    return ax.scatter(
        x[self.n_depot :],
        y[self.n_depot :],
        alpha=alpha[self.n_depot :],
        **vertex_kwargs,
    )


def plot_VRPEnv(
    self: envs.vrp.VRPEnv,
    *,
    ax_dict: Union[Dict[str, matplotlib.axes.Axes], None] = None,
    **kwargs: Any,
) -> Any:
    if ax_dict is None:
        ax_dict = dict()

    if ax_dict:
        try:
            vrp_ax = ax_dict["vrp_ax"]
            demand_ax = ax_dict["demand_ax"]
            loading_ax = ax_dict["loading_ax"]
        except KeyError as err:
            raise ValueError("All axes should be all given or none at all") from err
    else:
        fig = plt.gcf()
        grid = fig.add_gridspec(
            2, 2, hspace=0, wspace=0, width_ratios=[4, 1], height_ratios=[1, 4]
        )
        vrp_ax = fig.add_subplot(grid[1, 0])
        demand_ax = fig.add_subplot(grid[0, 0], sharex=vrp_ax)
        loading_ax = fig.add_subplot(grid[1, 1], sharey=vrp_ax)

    # keyword arguments
    vrp_kwargs = _filter_dict(kwargs, "vrp_", ignore={"vrp_alpha"})
    location_kwargs = _filter_dict(kwargs, "location_")
    demand_kwargs = _filter_dict(kwargs, "demand_")
    loading_kwargs = _filter_dict(kwargs, "loading_")
    if len(kwargs):
        raise ValueError(f"{set(kwargs)} not needed")

    vrp_art = plot_VRP(self.vrp, ax=vrp_ax, **vrp_kwargs)
    x = [self.vrp.vertices[l][0] for l in self.locations]
    y = [self.vrp.vertices[l][1] for l in self.locations]
    vrp_ax.scatter(x, y, alpha=1, **location_kwargs)
    demand_art = demand_ax.bar(
        np.arange(self.vrp.n_vertex), self.vrp.demand, **demand_kwargs
    )
    demand_ax.spines["top"].set_visible(False)
    demand_ax.spines["right"].set_visible(False)
    demand_ax.xaxis.set_visible(False)
    demand_ax.set_ylabel("Demand")
    base = loading_ax.transData
    rot = transforms.Affine2D().rotate_deg(90).scale(-1, 1)
    loading_art = loading_ax.bar(
        np.arange(self.vrp.n_vehicle),
        self.loading,
        transform=rot + base,
        **loading_kwargs,
    )
    loading_ax.spines["top"].set_visible(False)
    loading_ax.spines["right"].set_visible(False)
    loading_ax.yaxis.set_visible(False)
    loading_ax.set_xlabel("Loading")

    return vrp_art, demand_art, loading_art


def plot_FMP(self: envs.fmp.FMP, *, ax: matplotlib.axes.Axes, **kwargs) -> Any:
    x, y = list(), list()
    for v in self.vertices:
        x.append(v.x)
        y.append(v.y)

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(1, 0, 0)

    vertex_kwargs = _filter_dict(kwargs, "vertex_")
    edge_kwargs = _filter_dict(kwargs, "edge_")
    if len(kwargs):
        raise ValueError(f"{set(kwargs)} not needed")

    # avoid alpha=0 when no demand
    # if max(self.demand) == 0:
    #     alpha = np.ones(self.demand.shape) * 0.6
    # else:
    #     alpha = self.demand / max(self.demand) * 0.4 + np.ones(self.demand.shape) * 0.6

    for e in self.edges:
        l1, l2 = self.vertices[e.start], self.vertices[e.end]
        ax.plot([l1.x, l2.x], [l1.y, l2.y], **edge_kwargs)

    # ax.scatter(x[: self.n_depot], y[: self.n_depot], alpha=1, **depot_kwargs)
    return ax.scatter(
        x,
        y,
        **vertex_kwargs,
    )


def plot_FMPEnv(
    self: envs.fmp.FMPEnv,
    *,
    ax_dict: Union[Dict[str, matplotlib.axes.Axes], None] = None,
    **kwargs: Any,
) -> Any:
    if ax_dict is None:
        ax_dict = dict()

    if ax_dict:
        try:
            fmp_ax = ax_dict["fmp_ax"]
            demand_ax = ax_dict["demand_ax"]
            loading_ax = ax_dict["loading_ax"]
        except KeyError as err:
            raise ValueError("All axes should be all given or none at all") from err
    else:
        fig = plt.gcf()
        grid = fig.add_gridspec(
            2, 2, hspace=0, wspace=0, width_ratios=[4, 1], height_ratios=[1, 4]
        )
        fmp_ax = fig.add_subplot(grid[1, 0])
        demand_ax = fig.add_subplot(grid[0, 0], sharex=fmp_ax)
        loading_ax = fig.add_subplot(grid[1, 1], sharey=fmp_ax)

    # keyword arguments
    fmp_kwargs = _filter_dict(kwargs, "fmp_", ignore={"fmp_alpha"})
    location_kwargs = _filter_dict(kwargs, "location_")
    # demand_kwargs = _filter_dict(kwargs, "demand_")
    # loading_kwargs = _filter_dict(kwargs, "loading_")
    # if len(kwargs):
    #     raise ValueError(f"{set(kwargs)} not needed")

    fmp_art = plot_FMP(self.fmp, ax=fmp_ax, **fmp_kwargs)
    x = [self.fmp.vertices[s.location].x for s in self.states]
    y = [self.fmp.vertices[s.location].y for s in self.states]
    fmp_ax.scatter(x, y, alpha=1, **location_kwargs)
    # demand_art = demand_ax.bar(
    #     np.arange(self.fmp.n_vertex), self.fmp.demand, **demand_kwargs
    # )
    demand_ax.spines["top"].set_visible(False)
    demand_ax.spines["right"].set_visible(False)
    demand_ax.xaxis.set_visible(False)
    demand_ax.set_ylabel("Demand")
    # base = loading_ax.transData
    # rot = transforms.Affine2D().rotate_deg(90).scale(-1, 1)
    # loading_art = loading_ax.bar(
    #     np.arange(self.fmp.n_vehicle),
    #     self.loading,
    #     transform=rot + base,
    #     **loading_kwargs,
    # )
    loading_ax.spines["top"].set_visible(False)
    loading_ax.spines["right"].set_visible(False)
    loading_ax.yaxis.set_visible(False)
    loading_ax.set_xlabel("Loading")
    return fmp_art
