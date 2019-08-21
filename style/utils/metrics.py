import numpy as np


def cross_entropy(dist, target_dist, epsilon=1e-12):
    dist = np.clip(dist, epsilon, 1.)
    N = dist.shape[0]
    ce = -np.sum(target_dist * np.log(dist)) / N
    return ce
