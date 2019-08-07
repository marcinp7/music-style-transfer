import numpy as np

import mido

from py_utils.math import normalize_dist
from py_utils.metrics import cross_entropy
from py_utils import group_by

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
interval2note = dict(enumerate(note_names))
note2interval = {note: interval for interval, note in interval2note.items()}

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

    def __repr__(self):
        return f'{self.name} mode'


def create_mode(mode, shift):
    intervals = mode.intervals
    return Mode(intervals[shift:] + intervals[:shift], shift)


major_mode = Mode([2, 2, 1, 2, 2, 2, 1])
minor_mode = create_mode(major_mode, shift=-2)
all_modes = [create_mode(major_mode, shift) for shift in range(len(Mode.names))]


def get_notes_dist(info, nchannel):
    note2time = group_by(nchannel['notes'], attr='key', func=lambda xs: sum(
        x.duration * x.velocity for x in xs))
    note2time = {note: mido.tick2second(
        time, info['ticks_per_beat'], info['tempo']) for note, time in note2time.items()}
    note2time['instrument'] = nchannel['instrument_name']
    return note2time


major_dist = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
major_dist = major_dist[major_mode.absolute_intervals]
normalize_dist(major_dist)

minor_dist = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
minor_dist = minor_dist[minor_mode.absolute_intervals]
normalize_dist(minor_dist)

target_mode_dist = (major_dist + minor_dist) / 2


def get_scales(note2time=None, notes_dist=None, modes=None, degrees=None):
    modes = modes or all_modes
    degrees = degrees or list(range(1, 8))
    if notes_dist is None:
        notes_dist = np.array([note2time.get(note, 0) for note in note_names])
        normalize_dist(notes_dist)

    degrees = [d - 1 for d in degrees]
    target_dist = target_mode_dist[degrees]
    normalize_dist(target_dist)

    data = []
    for i, note in enumerate(note_names):
        for mode in modes:
            intervals = (np.asarray(mode.absolute_intervals) + i) % 12
            sample_dist = notes_dist[intervals]
            coverage = sample_dist.sum()
            sample_dist = sample_dist[degrees]
            normalize_dist(sample_dist)
            data.append({
                'coverage': coverage,
                'tonic': note,
                'mode': mode,
                'cross_entropy': cross_entropy(sample_dist, target_dist),
                'dist': sample_dist,
            })

    for d in data:
        d['loss'] = d['cross_entropy'] * (2 - d['coverage'])

    return data


def get_scale(*args, **kwargs):
    scales = get_scales(*args, **kwargs)
    scale = min(scales, key=lambda x: x['loss'])
    return scale
