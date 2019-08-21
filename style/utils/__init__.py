from collections import defaultdict
import itertools


def freeze(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(map(freeze, obj))
    if isinstance(obj, set):
        return frozenset(obj)
    return obj


def group_by(data, key=None, attr=None, func=None, save_indices=False):
    if not callable(key):
        if key:
            key_name = key

            def key(x):
                return x[key_name]
        elif attr:
            def key(x):
                return getattr(x, attr)
        else:
            key = None

    key2elems = defaultdict(list)
    for i, d in enumerate(data):
        k = freeze(key(d)) if key is not None else d
        key2elems[k].append(i if save_indices else d)

    if func:
        return {key: func(elems) for key, elems in key2elems.items()}
    return dict(key2elems)


def flatten(elems):
    return list(itertools.chain(*elems))
