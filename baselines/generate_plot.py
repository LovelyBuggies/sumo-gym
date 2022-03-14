import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

import json


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def plot_loss(lower_loss_file, upper_loss_file, upper_loss_key, upper_smooth, start_index, end_index, truncate_prefix):
    lower_loss_dict = json.load(open(lower_loss_file, "r"))
    episode = np.array([int(key) for key, _ in lower_loss_dict.items()][start_index:end_index])
    lower_loss = np.array([value["total_mean"] for _, value in lower_loss_dict.items()][start_index:end_index])

    upper_loss_dict = json.load(open(upper_loss_file, "r"))
    upper_loss = np.array([value[upper_loss_key] for _, value in upper_loss_dict.items()][start_index:end_index])

    plt.rcParams["font.size"] = '14'
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    y1 = ax1.plot(episode[truncate_prefix:], lower_loss[truncate_prefix:], color="navy", ls='--', lw=4, label="lower level loss")

    upper_loss_smooth = savitzky_golay(upper_loss, 51, 3) if upper_smooth else upper_loss
    y2 = ax2.plot(episode[truncate_prefix:], upper_loss_smooth[truncate_prefix:], color="indianred", ls='-', lw=4, label="upper level loss")

    lns = y1 + y2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right")


def plot_reward(lower_reward_file, upper_reward_file, upper_reward_key, smooth, start_index, end_index, scale=False):
    lower_reward_dict = json.load(open(lower_reward_file, "r"))
    episode = np.array([int(key) for key, _ in lower_reward_dict.items()][start_index:end_index])
    lower_reward = np.array([value*int(key) if scale else value for key, value in lower_reward_dict.items()][start_index:end_index])

    upper_reward_dict = json.load(open(upper_reward_file, "r"))
    upper_reward = np.array([value[upper_reward_key] for _, value in upper_reward_dict.items()][start_index:end_index])

    plt.rcParams["font.size"] = '14'
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    lower_reward_smooth = savitzky_golay(lower_reward, 51, 5) if smooth else savitzky_golay(lower_reward, 31, 1)
    y1 = ax1.plot(episode, lower_reward_smooth, color="navy", ls='--', lw=4, label="lower level reward")

    upper_reward_smooth = savitzky_golay(upper_reward, 51, 5) if smooth else savitzky_golay(upper_reward, 11, 1)  
    y2 = ax2.plot(episode, upper_reward_smooth, color="indianred", ls='-', lw=4, label="upper level reward")

    plt.xlabel("episode")

    lns = y1 + y2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right")

plot_loss("jumbo-lower-loss.json", "jumbo-upper-loss.json", 'v0', True, 50, 129, 10)
plt.savefig("jumbo-loss.pdf")
plot_reward("jumbo-lower-reward.json", "jumbo-upper-reward.json", '0', True, 0, 57, True)
plt.savefig("jumbo-reward.pdf")
plot_loss("cosmos-lower-loss.json", "cosmos-upper-loss.json", '-5', False, 0, 17, 4)
plt.savefig("cosmos-loss.pdf")
plot_reward("cosmos-lower-reward.json", "cosmos-upper-reward.json", '-2', False, 0, 17)
plt.savefig("cosmos-reward.pdf")

