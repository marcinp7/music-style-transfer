import numpy as np


def normalize_dist(dist):
    dist = np.array(dist)
    assert len(dist)
    total_prob = dist.sum()
    if total_prob > 0:
        return dist / total_prob
    dist[:] = 1
    return normalize_dist(dist)


def round_number(number, precision=1):
    remainder_pos = number % precision
    remainder_neg = abs(remainder_pos - precision)
    if remainder_pos < remainder_neg:
        return number - remainder_pos, remainder_pos
    return number + remainder_neg, -remainder_neg
