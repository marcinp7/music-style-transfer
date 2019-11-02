from collections import defaultdict
import glob
import itertools
import math
import os

from tqdm import tqdm


def iter_all_files(path, pattern='**/*'):
    pattern = os.path.join(path, pattern)
    results = glob.iglob(pattern)
    files = (x for x in results if os.path.isfile(x))
    yield from files


class ProgressBar:
    def __init__(self, n_iterations=None, momentum=.99, biased=False, show_min_for=(),
                 show_max_for=()):
        self.n_iterations = n_iterations
        self.momentum = momentum
        self.biased = biased
        self.show_min_for = show_min_for
        self.show_max_for = show_max_for

        self.pbar = tqdm(total=self.n_iterations)
        self.clear_values()

        self.decimal_places = 2

    def clear_values(self):
        self.values_sum = defaultdict(int)
        self.values_seen = defaultdict(int)
        self.min_values = {}
        self.max_values = {}
        self.avg_values = {}

    def initial_values(self, **values):
        self.avg_values.update(values)
        self.biased = True

    def add(self, n, **values):
        self.pbar.update(n)
        self.update_values(n, **values)

        if self.pbar.n == self.n_iterations:
            self.close()

    def update_values(self, n, **values):
        values = {k: v for k, v in values.items() if v is not None}
        if self.biased:
            avg_values = {k: self.avg_values.get(k, 0)*self.momentum + v*(1-self.momentum)
                          for k, v in values.items()}
            self.avg_values.update(avg_values)
        else:
            for k, v in values.items():
                self.values_sum[k] = self.values_sum[k]*self.momentum + v*n
                self.values_seen[k] = self.values_seen[k]*self.momentum + n
            self.avg_values = {k: self.values_sum[k] / self.values_seen[k] for k in self.values_sum}
        self.min_values = {k: min(avg, self.min_values.get(k) or math.inf)
                           for k, avg in self.avg_values.items()}
        self.max_values = {k: max(avg, self.max_values.get(k) or -math.inf)
                           for k, avg in self.avg_values.items()}

        description = [f'{k}: {v:.{self.decimal_places}f}' for k, v in self.avg_values.items()]
        description += [f'min {k}: {v:.{self.decimal_places}f}'
                        for k, v in self.min_values.items() if k in self.show_min_for]
        description += [f'max {k}: {v:.{self.decimal_places}f}'
                        for k, v in self.max_values.items() if k in self.show_max_for]
        self.pbar.set_postfix_str(', '.join(description))

    def close(self):
        self.pbar.close()

    def __getitem__(self, k):
        return self.avg_values[k]

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.close()


def freeze(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
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


def dict_map(func, d, recursive=False):
    if not recursive:
        return {k: func(v) for k, v in d.items()}
    if isinstance(d, dict):
        return {k: dict_map(func, v, recursive) for k, v in d.items()}
    return func(d)


def make_dirs(path):
    path = path or '.'
    os.makedirs(path, exist_ok=True)


def assert_dir(path):
    make_dirs(os.path.dirname(path))
