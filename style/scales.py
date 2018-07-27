import numpy as np

from py_utils.math import cross_entropy, normalize_dist

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
semitones2note = dict(enumerate(notes))

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

    def __repr__(self):
        return f'{self.name} mode (intervals: {self.intervals}, chords: {self.chords})'


def create_mode(mode, shift):
    intervals = mode.intervals
    return Mode(intervals[shift:] + intervals[:shift], shift)


major_mode = Mode([2, 2, 1, 2, 2, 2, 1])
minor_mode = create_mode(major_mode, shift=-2)
all_modes = [create_mode(major_mode, shift) for shift in range(len(Mode.names))]

target_mode_dist = np.array([0.21, 0.11, 0.16, 0.13, 0.16, 0.13, 0.10])


def get_scales(note2time=None, sound_dist=None, modes=None, degrees=None):
    modes = modes or all_modes
    degrees = degrees or list(range(1, 8))
    if sound_dist is None:
        sound_dist = np.array([note2time.get(note, 0) for note in notes])
        sound_dist /= sound_dist.sum()

    degrees = [d - 1 for d in degrees]
    target_dist = target_mode_dist[degrees]
    normalize_dist(target_dist)

    data = []
    for i, note in enumerate(notes):
        for mode in modes:
            intervals = (np.asarray(mode.absolute_intervals) + i) % 12
            sample_dist = sound_dist[intervals]
            coverage = sample_dist.sum()
            sample_dist = sample_dist[degrees]
            normalize_dist(sample_dist)
            data.append({
                'coverage': coverage,
                'tonic': note,
                'mode': mode.name,
                'cross_entropy': cross_entropy(sample_dist, target_dist),
                'dist': sample_dist,
            })
    for d in data:
        d['loss'] = d['cross_entropy'] * (1 - d['coverage'])

    return data


def get_scale(*args, **kwargs):
    scales = get_scales(*args, **kwargs)
    scale = min(scales, key=lambda x: x['loss'])
    return scale
