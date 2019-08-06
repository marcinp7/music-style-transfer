from collections import defaultdict
from copy import copy, deepcopy
from fractions import Fraction
import math

import numpy as np

import mido
from mido import Message, MidiFile

from py_utils import group_by
from py_utils.data import list2df
from py_utils.math import round_number, normalize_dist

from style.midi import (
    get_instrument_id,
    program2instrument,
    is_pitched,
    default_tempo,
    default_volume,
    max_volume,
    max_velocity,
)
from style.scales import interval2note, note2interval, get_notes_dist, notes, get_scale
from style.exceptions import MidiFormatError


def merge_channels(channels):
    merged_channel = copy(channels[0])
    merged_channel['messages'] = []
    channel_idx = merged_channel['index']
    for channel in channels:
        for msg in channel['messages']:
            msg = copy(msg)
            msg.channel = channel_idx
            merged_channel['messages'].append(msg)
    merged_channel['messages'] = sorted(
        merged_channel['messages'], key=lambda x: x.time)
    return merged_channel


def get_channel_info(channel):
    channel_info = {k: v for k, v in channel.items() if k != 'messages'}
    return channel_info


def note_id2note(note_id, pitched=True):
    if pitched:
        octave, interval = divmod(note_id, 12)
        return {
            'note': interval2note[interval],
            'octave': octave - 1
        }
    return {
        'note': str(note_id),
        'octave': None,
    }


def note2note_id(note, pitched=True):
    if pitched:
        interval = note2interval[note['note']]
        return 12 * (note['octave'] + 1) + interval
    return int(note['note'])


def merge_tracks(tracks):
    """Merge all events to a single track and apply global timing."""
    msgs = []
    for track in tracks:
        time = 0
        for msg in track:
            msg = copy(msg)
            time += msg.time
            msg.time = time
            msgs.append(msg)
    msgs = sorted(msgs, key=lambda x: x.time)
    return list(msgs)


def split_channels(mid, max_time=1e6):
    def check(condition, error_message):
        if not condition:
            raise MidiFormatError(error_message)

    info = {
        'ticks_per_beat': mid.ticks_per_beat,
        'bpm': [],
        'time_signature': {
            'numerator': 4,
            'denominator': 4,
            'value': 1.,
        },
        'key': None,
    }
    channels = defaultdict(
        lambda: {'messages': [], 'program': 0})
    played_channels = set()
    modified_channels = set()

    messages_to_ignore = [
        'smpte_offset',
        'midi_port',
        'sysex',
        'end_of_track',
        'track_name',
        'copyright',
        'lyrics',
        'marker',
        'sequencer_specific',
        'channel_prefix',
        'text',
        'instrument_name',
        'aftertouch',
        'polytouch',
        'cue_marker',
        'unknown_meta',
        'sequence_number',
        'control_change',
    ]

    tempo = default_tempo
    tempo_change_time = 0
    tempo2total_time = defaultdict(int)

    for msg in merge_tracks(mid.tracks):
        if msg.type in messages_to_ignore:
            continue
        check(msg.time <= max_time, "MIDI file too long")
        msg = copy(msg)

        if msg.type == 'time_signature':
            ts = info['time_signature'] = {
                'numerator': msg.numerator,
                'denominator': msg.denominator,
                'value': msg.numerator / msg.denominator
            }
            check(not played_channels or ts ==
                  info['time_signature'], "Time signature changed")
        elif msg.type == 'key_signature':
            check(not played_channels or info['key']
                  == msg.key, "Key signature changed")
            info['key'] = msg.key
        elif msg.type == 'set_tempo':
            if msg.tempo != tempo:
                tempo2total_time[tempo] += msg.time - tempo_change_time
                tempo = msg.tempo
                tempo_change_time = msg.time
        elif msg.type == 'program_change':
            if (channels[msg.channel]['program'] != msg.program and
                    msg.channel in played_channels):
                modified_channels.add(msg.channel)
            channels[msg.channel]['program'] = msg.program
        elif msg.type in ['note_on', 'note_off', 'pitchwheel']:
            if msg.type == 'note_on':
                if msg.velocity == 0:
                    msg = Message('note_off', channel=msg.channel, note=msg.note, velocity=0,
                                  time=msg.time)
                else:
                    check(msg.channel not in modified_channels,
                          f"Channel {msg.channel} settings changed")
                    played_channels.add(msg.channel)
            channels[msg.channel]['messages'].append(msg)
        else:
            raise MidiFormatError(f"Unknown message type: {msg.type}")

    tempo2total_time[tempo] += msg.time - tempo_change_time
    info['duration'] = msg.time
    info['ticks_per_bar'] = int(
        mid.ticks_per_beat * 4 * info['time_signature']['value'])

    tempo2total_time = {k: v for k, v in tempo2total_time.items() if v}
    info['tempo2time'] = tempo2total_time
    info['tempo'] = max(tempo2total_time.items(), key=lambda x: x[1])[0]
    info['bpm'] = int(mido.tempo2bpm(info['tempo']))
    info['n_bars'] = info['duration'] / info['ticks_per_bar']
    info['n_beats'] = info['time_signature']['numerator']

    for k, v in channels.items():
        v['index'] = k

    channels = sorted(channels.values(), key=lambda x: x['index'])
    channels = [channel for channel in channels
                if any(msg.type == 'note_on' for msg in channel['messages'])]

    for channel in channels:
        channel['instrument_id'] = get_instrument_id(
            channel['program'], channel['index'])
        channel['instrument_name'] = program2instrument[channel['instrument_id']]

    return channels, info


def note2scale_loc(note, mode, tonic):
    tonic_interval = note2interval[tonic]
    interval = note2interval[note['note']] - tonic_interval
    degree = mode.get_degree(interval)
    sharp = not isinstance(degree, int)
    degree = int(degree)
    octave = note['octave']
    if interval < 0:
        octave -= 1
    return dict(
        octave=octave,
        degree=degree,
        sharp=sharp,
    )


def scale_loc2note(octave, degree, mode, tonic, sharp=False):
    tonic_interval = note2interval[tonic]
    interval = mode.absolute_intervals[degree - 1] + tonic_interval
    if sharp:
        interval += 1
    if interval >= 12:
        octave += 1
        interval -= 12
    return dict(
        note=interval2note[interval],
        octave=octave,
    )


class ChannelConverter:
    def __init__(self, info, beat_divisors=(8, 3), n_octaves=8, min_percussion=35,
                 max_percussion=81):
        self.info = info
        self.beat_divisors = beat_divisors
        self.n_octaves = n_octaves
        self.min_percussion = min_percussion
        self.max_percussion = max_percussion

        self.beat_fractions = sorted({
            Fraction(i, divisor)
            for divisor in self.beat_divisors
            for i in range(divisor)
        })
        self.beat_fraction2idx = {fraction: i for i,
                                  fraction in enumerate(self.beat_fractions)}

        self.n_notes = self.n_octaves * 7
        self.n_unpitched = self.max_percussion - self.min_percussion + 1
        self.n_note_features = 3  # duration, velocity, sharp
        self.n_unpitched_features = 2

    def channel2nchannel(self, channel):
        nchannel = {k: deepcopy(v)
                    for k, v in channel.items() if k != 'messages'}
        nchannel['notes'] = []
        pitched = is_pitched(channel['instrument_id'])
        channel_volume = default_volume

        note_id2last_played_note = {}
        for msg in channel['messages']:
            if msg.type in ['note_on', 'note_off']:
                note_id = msg.note
                if note_id in note_id2last_played_note:
                    note = note_id2last_played_note[note_id]
                    note['end_time'] = msg.time
                    del note_id2last_played_note[note_id]
                if msg.type == 'note_on':
                    note = note_id2note(note_id, pitched)
                    # todo: keep track of velocity when converting back to the list of messages
                    velocity = msg.velocity * channel_volume / (max_velocity * max_volume)
                    note.update(
                        note_id=note_id,
                        velocity=velocity,
                        time=msg.time,
                        end_time=msg.time,
                        time_sec=mido.tick2second(
                            msg.time,
                            self.info['ticks_per_beat'],
                            self.info['tempo']
                        )
                    )
                    nchannel['notes'].append(note)
                    note_id2last_played_note[note_id] = note
            elif msg.type == 'control_change' and msg.control == 7:
                channel_volume = msg.value

        for note in nchannel['notes']:
            note['duration'] = note['end_time'] - note['time']

        return nchannel

    def nchannel2kchannel(self, nchannel, in_place=False):
        kchannel = nchannel if in_place else deepcopy(nchannel)
        pitched = is_pitched(nchannel['instrument_id'])
        for note in kchannel['notes']:
            if pitched:
                scale_loc = note2scale_loc(note, self.mode, self.tonic)
            else:
                scale_loc = dict(
                    octave=None,
                    degree=None,
                    sharp=None
                )
            note.update(
                scale_octave=scale_loc['octave'],
                scale_degree=scale_loc['degree'],
                sharp=scale_loc['sharp'],
            )
        return kchannel

    def kchannel2qchannel(self, kchannel, in_place=False):
        def ticks2loc(ticks):
            bar, ticks = divmod(ticks, self.info['ticks_per_bar'])
            beat, ticks = divmod(ticks, self.info['ticks_per_beat'])

            return bar, beat, ticks

        divisor2ticks = {divisor: self.info['ticks_per_beat'] / divisor
                         for divisor in self.beat_divisors}

        def note_quantizations(time, duration=None):
            # todo: use numpy
            for divisor, ticks in divisor2ticks.items():
                qtime, time_error = round_number(time, ticks)
                total_error = abs(time_error)
                yield (qtime, divisor), total_error

        qchannel = kchannel if in_place else deepcopy(kchannel)

        for note in qchannel['notes']:
            time = note['time']
            duration = note.get('duration')
            qtime, divisor = min(note_quantizations(
                time, duration), key=lambda x: x[1])[0]
            note['qtime'] = int(qtime)
            note['qduration'] = note['end_time'] - note['qtime']

            bar, beat, ticks = ticks2loc(note['qtime'])
            quants = int(ticks // divisor2ticks[divisor])
            note.update(
                bar=int(bar),
                beat=int(beat),
                beat_fraction=Fraction(quants, divisor),
            )

        return qchannel

    def qchannel2channel(self, channel_info, qchannel):
        def loc2ticks(bar, beat, beat_fraction):
            return (
                bar * self.info['ticks_per_bar'] +
                beat * self.info['ticks_per_beat'] +
                int(beat_fraction * self.info['ticks_per_beat'])
            )

        pitched = is_pitched(channel_info['instrument_id'])
        messages = []
        channel_idx = channel_info['index']
        for note in qchannel['notes']:
            note = copy(note)
            if pitched:
                note_ = scale_loc2note(
                    note['scale_octave'],
                    note['scale_degree'],
                    self.mode,
                    self.tonic,
                    note['sharp']
                )
                note.update(note_)
            note_id = note2note_id(note, pitched)
            time = loc2ticks(note['bar'], note['beat'], note['beat_fraction'])

            note_on = Message('note_on', channel=channel_idx,
                              note=note_id, velocity=note['velocity'], time=time)
            note_off = Message('note_off', channel=channel_idx,
                               note=note_id, time=time+note['qduration'])
            messages += [note_on, note_off]

        channel = deepcopy(channel_info)
        channel['messages'] = sorted(messages, key=lambda x: x.time)
        return channel

    def qchannel2vchannel(self, qchannel):
        pitched = is_pitched(qchannel['instrument_id'])
        bars = [self.get_empty_bar(pitched) for _ in range(self.n_bars)]
        for note in qchannel['notes']:
            try:
                note_idx = self.note2idx(note, pitched)
            except ValueError:
                continue

            partial_beat = self.get_empty_beat(pitched)
            features = [note['qduration'], note['velocity'] / 127]
            if pitched:
                features.append(note['sharp'])
            partial_beat[self.beat_fraction2idx[note['beat_fraction']]
                         ][note_idx] = features

            bar = bars[note['bar']]
            bar[note['beat']] = np.maximum(bar[note['beat']], partial_beat)
        return bars

    def vchannel2qchannel(self, channel_info, vchannel):
        pitched = is_pitched(channel_info['instrument_id'])
        qchannel = copy(channel_info)
        qchannel['notes'] = []
        for bar_idx, bar in enumerate(vchannel):
            for beat_idx, beat in enumerate(bar):
                for fraction, vnotes in zip(self.beat_fractions, beat):
                    velocities = vnotes[:, 1]
                    note_inds = np.nonzero(velocities)[0]
                    for note_idx in note_inds:
                        vnote = vnotes[note_idx]
                        if pitched:
                            duration, velocity, sharp = vnote
                        else:
                            duration, velocity = vnote
                            sharp = False
                        note = self.idx2note(note_idx, pitched, sharp)
                        note.update(
                            bar=bar_idx,
                            beat=beat_idx,
                            beat_fraction=fraction,
                            qduration=int(duration),
                            velocity=int(velocity * 127),
                        )
                        qchannel['notes'].append(note)
        return qchannel

    @property
    def mode(self):
        return self.info['scale']['mode']

    @property
    def tonic(self):
        return self.info['scale']['tonic']

    @property
    def n_bars(self):
        return math.ceil(self.info['n_bars'])

    def n_features(self, pitched):
        return self.n_note_features if pitched else self.n_unpitched_features

    def get_empty_beat(self, pitched):
        n_notes = self.n_notes if pitched else self.n_unpitched
        return np.zeros([len(self.beat_fractions), n_notes, self.n_features(pitched)])

    def get_empty_bar(self, pitched):
        return [self.get_empty_beat(pitched) for _ in range(self.info['n_beats'])]

    def note2idx(self, note, pitched):
        if pitched:
            octave = note['scale_octave']
            degree = note['scale_degree']
            note_idx = octave * 7 + (degree - 1)
            if note_idx < 0 or note_idx >= self.n_notes:
                raise ValueError()
            return note_idx
        else:
            note_idx = int(note['note'])
            if note_idx < self.min_percussion or note_idx > self.max_percussion:
                raise ValueError()
            note_idx -= self.min_percussion
            return note_idx

    def idx2note(self, note_idx, pitched, sharp=False):
        if pitched:
            degree = note_idx % 7 + 1
            note_idx -= degree - 1
            octave = note_idx // 7
            return dict(
                scale_degree=degree,
                scale_octave=octave,
                sharp=sharp,
            )
        else:
            return dict(
                note=note_idx+self.min_percussion,
            )


# todo: merge the same instruments, remember the volume
def merge_channels_by_instrument(channels):
    channels_grouped = group_by(
        channels, 'instrument_name', func=merge_channels)
    channels_grouped = list(channels_grouped.values())
    return channels_grouped


def get_input(filename):
    try:
        mid = MidiFile(filename)
    except (OSError, ValueError, KeyError, EOFError):
        raise MidiFormatError('Error loading MIDI file')
    channels, info = split_channels(mid)
    channels = merge_channels_by_instrument(channels)
    channels_info = [get_channel_info(channel) for channel in channels]

    cc = ChannelConverter(info)
    nchannels = [cc.channel2nchannel(channel) for channel in channels]

    notes_dist_per_instrument = [get_notes_dist(
        info, nchannel) for nchannel in nchannels]
    notes_dist = list2df(notes_dist_per_instrument).reindex(
        columns=notes).sum()
    notes_dist = np.asarray(notes_dist)
    normalize_dist(notes_dist)

    scale = get_scale(notes_dist=notes_dist)
    info['scale'] = scale

    kchannels = [cc.nchannel2kchannel(nchannel) for nchannel in nchannels]
    qchannels = [cc.kchannel2qchannel(kchannel) for kchannel in kchannels]
    vchannels = [cc.qchannel2vchannel(qchannel) for qchannel in qchannels]

    return info, channels_info, vchannels


def get_inputs(files, shuffle=False):
    if shuffle:
        files = files[:]
        np.random.shuffle(files)
    for file in files:
        yield get_input(file)
