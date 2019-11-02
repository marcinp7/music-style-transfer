import numpy as np


def to_1d_array(obj, copy=True):
    obj_with_dummy = [0, *obj]
    array = np.array(obj_with_dummy, dtype='object', copy=copy)
    return array[1:]


def as_1d_array(obj):
    return to_1d_array(obj, copy=False)


def random_sample(*arrays, size, replace=False, return_indices=False):
    min_len = min(len(a) for a in arrays)
    if isinstance(size, float):
        size = int(size * min_len)
    inds = np.random.choice(range(min_len), size, replace=replace)
    arrays = [as_1d_array(a) for a in arrays]
    samples = [a[inds] for a in arrays]
    if return_indices:
        samples.append(inds)

    if len(samples) == 1:
        return samples[0]
    return tuple(samples)
