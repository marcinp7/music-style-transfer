from copy import copy, deepcopy
import math
import numpy as np
import os
import re

from collections import defaultdict
from fractions import Fraction
from py_utils import flatten, replace_none
from py_utils.math import round_number

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from style.scales import interval2note, note2interval

here = os.path.dirname(__file__)


def get_path(path):
    return os.path.join(here, path)


def parse_programs(path):
    program2instrument = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'([0-9]+) (.*)', line)
            if m:
                program, name = m.groups()
                program2instrument[int(program)-1] = name
    return program2instrument


program2instrument = parse_programs(get_path('midi_programs.txt'))
program2instrument[-1] = 'Percussion'


def get_instrument_id(program, channel=0):
    if channel == 9:
        return -1
    return program


def is_sound_effect(instrument_id):
    return instrument_id > 119


def is_pitched(instrument_id):
    return instrument_id >= 0 and not is_sound_effect(instrument_id)


def play_midi(mid, portname=None):
    with mido.open_output(portname) as output:
        try:
            for message in mid.play():
                output.send(message)

        except KeyboardInterrupt:
            output.reset()


def create_midi(info, *channels, max_delta_time=math.inf):
    max_delta_time = mido.second2tick(max_delta_time, info['ticks_per_beat'], info['tempo'])
    if math.isfinite(max_delta_time):
        max_delta_time = int(max_delta_time)

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    mid.ticks_per_beat = info['ticks_per_beat']

    ts = info['time_signature']
    track.append(MetaMessage('time_signature',
                             numerator=ts['numerator'], denominator=ts['denominator']))

    track.append(MetaMessage('set_tempo', tempo=info['tempo']))

    for channel in channels:
        if channel['index'] != 9:
            track.append(Message('program_change',
                                 channel=channel['index'], program=channel['program']))

        if 'volume' in channel:
            track.append(Message('control_change',
                                 channel=channel['index'], control=7, value=channel['volume']))

    msgs = flatten([channel['messages'] for channel in channels])
    msgs = sorted(msgs, key=lambda x: x.time)

    time = 0
    for msg in msgs:
        msg = copy(msg)
        delta_time = min(msg.time - time, max_delta_time)
        time = msg.time
        msg.time = delta_time
        track.append(msg)
    delta_time = min(info['duration'] - time, max_delta_time)
    track.append(MetaMessage('end_of_track', time=delta_time))

    return mid


def merge_tracks(tracks):
    msgs = []
    for track in tracks:
        time = 0
        for msg in track:
            msg = copy(msg)
            time += msg.time
            msg.time = time
            msgs.append(msg)
    msgs = sorted(msgs, key=lambda x: x.time)
    return msgs


def merge_channels(channels):
    merged_channel = copy(channels[0])
    merged_channel['messages'] = []
    channel_idx = merged_channel['index']
    for channel in channels:
        for msg in channel['messages']:
            msg = copy(msg)
            msg.channel = channel_idx
            merged_channel['messages'].append(msg)
    merged_channel['messages'] = sorted(merged_channel['messages'], key=lambda x: x.time)
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


# todo: allow multiple instruments per channel
@replace_none(([], None))
def split_channels(mid, max_time=1e6):
    info = {
        'ticks_per_beat': mid.ticks_per_beat,
        'bpm': [],
        'time_signature': {
            'numerator': 4,
            'denominator': 4,
            'value': 1.,
        }
    }
    channels = defaultdict(lambda: {'messages': [], 'program': 0, 'volume': 96})
    played_channels = set()
    non_playable_channels = set()

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
    ]

    tempo = None
    tempo_change_time = 0
    tempo2time = defaultdict(int)

    for msg in list(merge_tracks(mid.tracks)):
        if msg.time > max_time:
            return None
        msg = copy(msg)
        if msg.type in messages_to_ignore:
            continue

        if msg.type == 'time_signature':
            ts = info['time_signature'] = {
                'numerator': msg.numerator,
                'denominator': msg.denominator,
                'value': msg.numerator / msg.denominator
            }
            if ts != info['time_signature']:
                assert not played_channels
            info['time_signature'] = ts
        elif msg.type == 'key_signature':
            if played_channels and info.get('key') != msg.key:
                return None
            info['key'] = msg.key
        elif msg.type == 'set_tempo':
            if tempo:
                tempo2time[tempo] += msg.time - tempo_change_time
            tempo = msg.tempo
            tempo_change_time = msg.time
        elif msg.type == 'control_change':
            if msg.control == 7:
                if (channels[msg.channel]['volume'] != msg.value and
                        msg.channel in played_channels):
                    non_playable_channels.add(msg.channel)
                channels[msg.channel]['volume'] = msg.value
            if msg.control == 10:
                channels[msg.channel]['pan'] = msg.value
        elif msg.type == 'program_change':
            if (channels[msg.channel]['program'] != msg.program and
                    msg.channel in played_channels):
                non_playable_channels.add(msg.channel)
            channels[msg.channel]['program'] = msg.program
        elif msg.type in ['note_on', 'note_off', 'pitchwheel']:
            if msg.type == 'note_on' and msg.velocity == 0:
                msg = Message('note_off', channel=msg.channel, note=msg.note,
                              velocity=msg.velocity, time=msg.time)
            channels[msg.channel]['messages'].append(msg)
            if msg.type == 'note_on':
                if msg.channel in non_playable_channels:
                    return None
                played_channels.add(msg.channel)
        else:
            raise Exception(f'Unknown message type: {msg.type}')

    tempo2time[tempo] += msg.time - tempo_change_time
    info['duration'] = msg.time
    info['ticks_per_bar'] = int(mid.ticks_per_beat * 4 * info['time_signature']['value'])

    tempo2time = {k: v for k, v in tempo2time.items() if v}
    info['tempo2time'] = tempo2time
    info['tempo'] = max(tempo2time.items(), key=lambda x: x[1])[0]
    if info['tempo'] is None:
        return None
    info['bpm'] = int(mido.tempo2bpm(info['tempo']))
    info['n_bars'] = info['duration'] / info['ticks_per_bar']
    info['n_beats'] = info['time_signature']['numerator']

    for k, v in channels.items():
        v['index'] = k

    channels = sorted(channels.values(), key=lambda x: x['index'])
    channels = [channel for channel in channels
                if any(msg.type == 'note_on' for msg in channel['messages'])]

    for channel in channels:
        channel['instrument_id'] = get_instrument_id(channel['program'], channel['index'])
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
        self.beat_fraction2idx = {fraction: i for i, fraction in enumerate(self.beat_fractions)}

        self.n_notes = self.n_octaves * 7
        self.n_note_features = 3  # duration, velocity, sharp
        self.n_unpitched_features = 2

    def channel2nchannel(self, channel):
        nchannel = {k: deepcopy(v) for k, v in channel.items() if k != 'messages'}
        nchannel['notes'] = []
        pitched = is_pitched(channel['instrument_id'])

        pitch2note = {}
        for msg in channel['messages']:
            if msg.type in ['note_on', 'note_off']:
                pitch = msg.note
                if pitch in pitch2note:
                    note = pitch2note[pitch]
                    note['end_time'] = msg.time
                    del pitch2note[pitch]
                if msg.type == 'note_on':
                    note = note_id2note(pitch, pitched)
                    note.update(
                        pitch=pitch,
                        velocity=msg.velocity,
                        time=msg.time,
                        end_time=msg.time,
                        time_sec=mido.tick2second(
                            msg.time,
                            self.info['ticks_per_beat'],
                            self.info['tempo']
                        )
                    )
                    nchannel['notes'].append(note)
                    pitch2note[pitch] = note

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
            qtime, divisor = min(note_quantizations(time, duration), key=lambda x: x[1])[0]
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
            partial_beat[self.beat_fraction2idx[note['beat_fraction']]][note_idx] = features

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
        return np.zeros([len(self.beat_fractions), self.n_notes, self.n_features(pitched)])

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
