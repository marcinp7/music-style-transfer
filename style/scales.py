import numpy as np

from style.utils.math import normalize_dist
from style.utils.metrics import cross_entropy

key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
interval2key = dict(enumerate(key_names))
key2interval = {key: interval for interval, key in interval2key.items()}

intervals2chord = {
    (0, 4, 7): 'M',
    (0, 3, 7): 'm',
    (0, 3, 6): 'dim',
    (0, 4, 6): '♭5',
    (0, 4, 8): 'aug',
    (0, 2, 6): '♭5/3',
}


def get_chord_name(chord):
    name = intervals2chord.get(tuple(chord))
    if name is None:
        raise Exception(f'Unkown chord: {chord}')
    return name


class Mode:
    names = [
        'Ionian',
        'Dorian',
        'Phrygian',
        'Lydian',
        'Mixolydian',
        'Aeolian',
        'Locrian',
    ]
    name2shift = {name: i for i, name in enumerate(names)}
    shift2name = {i: name for i, name in enumerate(names)}

    def __init__(self, intervals, shift=0):
        self.intervals = intervals
        self.shift = shift

        self.tonic_intervals = [0]
        for interval in self.intervals:
            self.tonic_intervals.append(self.tonic_intervals[-1] + interval)

        assert len(self) == len(self.names)

        self.absolute_intervals = [0]
        for interval in self.intervals[:-1]:
            self.absolute_intervals.append(self.absolute_intervals[-1] + interval)

        self.interval2degree = {}
        for degree, interval in enumerate(self.absolute_intervals):
            self.interval2degree[interval] = degree + 1

        prev_degree = 1
        for interval in range(12):
            if interval in self.interval2degree:
                prev_degree = self.interval2degree[interval]
            else:
                self.interval2degree[interval] = prev_degree + .5

    @property
    def name(self):
        return self.shift2name[self.shift % len(self.names)]

    def __len__(self):
        return len(self.intervals)

    def get_tonic_interval(self, i):
        return self.tonic_intervals[i % len(self)]

    def get_chord(self, i):
        intervals = [self.get_tonic_interval(j) for j in [i, i+2, i+4]]
        tonic_interval = intervals[0]
        intervals = [(interval - tonic_interval) % 12 for interval in intervals]
        return get_chord_name(intervals)

    @property
    def chords(self):
        return [self.get_chord(i) for i in range(len(self))]

    def get_degree(self, interval):
        return self.interval2degree[interval % 12]

    def get_degree(self, interval):
        return self.interval2degree[interval % 12]

    def __repr__(self):
        return f'{self.name} mode'


def create_mode(mode, shift):
    intervals = mode.intervals
    return Mode(intervals[shift:] + intervals[:shift], shift)


def get_relative_degree(interval, source_scale, target_scale):
    relative_shift = (source_scale.shift - target_scale.shift) % 7
    relative_interval = target_scale.tonic_intervals[relative_shift]
    relative_degree = target_scale.get_degree(interval + relative_interval)
    return relative_degree


major_mode = Mode([2, 2, 1, 2, 2, 2, 1])
minor_mode = create_mode(major_mode, shift=-2)
all_modes = [create_mode(major_mode, shift) for shift in range(len(Mode.names))]

major_dist = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
major_dist = normalize_dist(major_dist)

minor_dist = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
minor_dist = normalize_dist(minor_dist)

target_mode_dist = (major_dist + minor_dist) / 2

major_intervals = major_mode.absolute_intervals
minor_intervals = minor_mode.absolute_intervals

# notes typically used in major in minor scales
typical_major_intervals = [0, 2, 4, 5, 6, 7, 9, 10, 11]
typical_minor_intervals = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11]


def get_all_modes(key2time=None, keys_dist=None, modes=None, degrees=None):
    modes = modes or all_modes
    degrees = degrees or list(range(1, 8))
    if keys_dist is None:
        keys_dist = np.array([key2time.get(key, 0) for key in key_names])
        normalize_dist(keys_dist)

    degrees = [d - 1 for d in degrees]
    target_dist = target_mode_dist[degrees]
    normalize_dist(target_dist)

    data = []
    for i, key in enumerate(key_names):
        for mode in modes:
            intervals = (np.asarray(mode.absolute_intervals) + i) % 12
            sample_dist = keys_dist[intervals]
            coverage = sample_dist.sum()
            sample_dist = sample_dist[degrees]
            normalize_dist(sample_dist)
            data.append({
                'coverage': coverage,
                'tonic': key,
                'mode': mode,
                'cross_entropy': cross_entropy(sample_dist, target_dist),
                'dist': sample_dist,
            })

    for d in data:
        d['loss'] = d['cross_entropy'] * (2 - d['coverage'])

    return data


def get_scales(key2time=None, keys_dist=None):
    if keys_dist is None:
        keys_dist = np.array([key2time.get(key, 0) for key in key_names])
        keys_dist = normalize_dist(keys_dist)

    major_key_scores = get_key_scores(
        keys_dist,
        major_dist,
        major_intervals,
        typical_major_intervals,
    )
    minor_key_scores = get_key_scores(
        keys_dist,
        minor_dist,
        minor_intervals,
        typical_minor_intervals,
    )

    data = []
    for key_score in major_key_scores:
        key_score['mode'] = 'major'
        data.append(key_score)
    for key_score in minor_key_scores:
        key_score['mode'] = 'minor'
        data.append(key_score)

    for d in data:
        # d['loss'] = d['cross_entropy'] * (1 - d['ndcg'])
        d['loss'] = d['cross_entropy'] * (1.5 - d['coverage']) * (2 - d['loose_coverage'])

    return data


def shift_intervals(intervals, shift):
    return (np.asarray(intervals) + shift) % 12


def get_key_scores(keys_dist, scale_dist, main_intervals, typical_intervals):
    for key in key_names:
        most_common = sorted(enumerate(-keys_dist), key=lambda x: x[1])
        most_common = [x[0] for x in most_common]
        most_common = sorted(enumerate(most_common), key=lambda x: x[1])
        most_common = [x[0] for x in most_common]
        from py_utils import metrics
        yield dict(
            key=key,
            coverage=keys_dist[main_intervals].sum(),
            loose_coverage=keys_dist[typical_intervals].sum(),
            cross_entropy=cross_entropy(keys_dist, scale_dist),
            ndcg=metrics.ndcg(scale_dist, most_common),
        )
        keys_dist = np.concatenate([keys_dist[1:], keys_dist[:1]])


def get_scale(*args, **kwargs):
    scales = get_scales(*args, **kwargs)
    scale = min(scales, key=lambda x: x['loss'])
    if scale['mode'] == 'major':
        scale['mode'] = major_mode
    elif scale['mode'] == 'minor':
        scale['mode'] = minor_mode
    return scale
