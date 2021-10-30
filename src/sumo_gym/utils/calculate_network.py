import numpy as np

def calculate_dist(p1, p2):
    return np.sqrt( np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2) )