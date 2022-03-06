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
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def plot_loss():
    loss_dict = json.load(open('loss.json', 'r'))
    episode = np.array([int(key) for key, _ in loss_dict.items()][10:950])
    v0_loss = np.array([value["v0"] for _, value in loss_dict.items()][10:950])
    v1_loss = np.array([value["v1"] for _, value in loss_dict.items()][10:950])
    v2_loss = np.array([value["v2"] for _, value in loss_dict.items()][10:950])

    # episode_new = np.linspace(episode.min(), episode.max(), 200) 

    # spl = make_interp_spline(episode, v0_reward, k=7)
    # v0_smooth = spl(episode_new)
    v0_smooth = savitzky_golay(v0_loss, 101, 5)
    plt.plot(episode, v0_smooth, color='aquamarine', label='v0 loss')

    # spl = make_interp_spline(episode, v1_loss, k=7)
    # v1_smooth = spl(episode_new)
    v1_smooth = savitzky_golay(v1_loss, 101, 5)
    plt.plot(episode, v1_smooth, color='cornflowerblue', label='v1 loss')

    # spl = make_interp_spline(episode, v2_loss, k=7)
    # v2_smooth = spl(episode_new)
    v2_smooth = savitzky_golay(v2_loss, 101, 5)
    plt.plot(episode, v2_smooth, color='wheat', label='v2 loss')

    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.show()

def plot_reward():
    reward_dict = json.load(open('reward.json', 'r'))
    episode = np.array([int(key) for key, _ in reward_dict.items()][10:950])
    v0_reward = np.array([value["v0"] for _, value in reward_dict.items()][10:950])
    v1_reward = np.array([value["v1"] for _, value in reward_dict.items()][10:950])
    v2_reward = np.array([value["v2"] for _, value in reward_dict.items()][10:950])

    # episode_new = np.linspace(episode.min(), episode.max(), 200) 

    # spl = make_interp_spline(episode, v0_reward, k=7)
    # v0_smooth = spl(episode_new)
    v0_smooth = savitzky_golay(v0_reward, 101, 5)
    plt.plot(episode, v0_smooth, color='aquamarine', label='v0 reward')


    # spl = make_interp_spline(episode, v1_loss, k=7)
    # v1_smooth = spl(episode_new)
    v1_smooth = savitzky_golay(v1_reward, 101, 5)
    plt.plot(episode, v1_smooth, color='cornflowerblue', label='v0 reward')

    # spl = make_interp_spline(episode, v2_loss, k=7)
    # v2_smooth = spl(episode_new)
    v2_smooth = savitzky_golay(v2_reward, 101, 5)
    plt.plot(episode, v2_smooth, color='wheat', label='v0 reward')

    # total_reward = v0_smooth + v1_smooth + v2_smooth
    # plt.plot(episode, total_reward, color='palevioletred', label='total reward')
   
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend(loc="upper right")
    plt.show()

plot_reward()
# plot_loss()