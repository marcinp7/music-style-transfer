from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
import math

import numpy as np
import mido

from style.utils import group_by, flatten
from style.utils.math import round_number

from style.midi import (
    get_instrument_id,
    program2instrument,
    is_pitched,
    default_tempo,
    default_volume,
    max_volume,
    max_velocity,
)
from style.scales import (
    interval2key,
    key2interval,
    get_relative_degree,
    major_mode,
)
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
    channel_info['pitched'] = is_pitched(channel_info['instrument_id'])
    return channel_info


def merge_tracks(tracks, apply_global_timing=False):
    msgs = []
    for track in tracks:
        time = 0
        for msg in track:
            msg = copy(msg)
            if apply_global_timing:
                time += msg.time
                msg.time = time
            msgs.append(msg)
    if apply_global_timing:
        msgs = sorted(msgs, key=lambda x: x.time)
    return list(msgs)


def split_channels(mid):
    global_messages = []
    channels = defaultdict(list)
    for msg in merge_tracks(mid.tracks, apply_global_timing=True):
        if hasattr(msg, 'channel'):
            channels[msg.channel].append(msg)
        else:
            global_messages.append(msg)

    return global_messages, list(channels.values())


messages_to_include = {
    'note_on',
    'note_off',
    'time_signature',
    'key_signature',
    'set_tempo',
    'program_change',
    'control_change',
    'pitchwheel',
}

messages_to_ignore = {
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
}


@dataclass
class NoteMessage:
    type: str
    note: int
    time: int
    velocity: float = 0


known_messages = messages_to_include | messages_to_ignore


def check(condition, error_message):
    if not condition:
        raise MidiFormatError(error_message)


def get_midi_info(global_messages, channels, ticks_per_beat):
    tempo = default_tempo
    tempo_change_time = 0
    tempo2total_time = defaultdict(int)

    channels_messages = flatten(channels)
    notes_on = filter(lambda x: x.type == 'note_on', channels_messages)
    notes_off = filter(lambda x: x.type == 'note_off', channels_messages)
    note_on_times = [n.time for n in notes_on]
    note_off_times = [n.time for n in notes_off]
    first_note_time, last_note_time = min(note_on_times), max(note_on_times)
    duration = max(note_on_times + note_off_times)

    def is_during_song(time):
        return first_note_time <= time <= last_note_time

    info = {
        'ticks_per_beat': ticks_per_beat,
        'duration': duration,
        'time_signature': {
            'numerator': 4,
            'denominator': 4,
            'value': 1.,
        },
        'key': None,
    }

    for msg in global_messages:
        if msg.type in messages_to_ignore:
            continue

        if msg.type == 'time_signature':
            ts = {
                'numerator': msg.numerator,
                'denominator': msg.denominator,
                'value': msg.numerator / msg.denominator
            }
            if ts != info['time_signature']:
                check(not is_during_song(msg.time), "Time signature changed")
                info['time_signature'] = ts
        elif msg.type == 'key_signature':
            if msg.key != info['key']:
                check(not is_during_song(msg.time), "Key signature changed")
                info['key'] = msg.key
        elif msg.type == 'set_tempo':
            if msg.tempo != tempo:
                tempo2total_time[tempo] += msg.time - tempo_change_time
                tempo = msg.tempo
                tempo_change_time = msg.time
        elif msg.type not in known_messages:
            raise MidiFormatError(f"Unknown message type: {msg.type}")

    info['ticks_per_bar'] = int(ticks_per_beat * 4 * info['time_signature']['value'])
    info['n_bars'] = duration / info['ticks_per_bar']
    info['n_beats'] = info['time_signature']['numerator']

    tempo2total_time[tempo] += duration - tempo_change_time
    tempo2total_time = {k: v for k, v in tempo2total_time.items() if v}
    info['tempo2time'] = tempo2total_time

    info['tempo'] = max(tempo2total_time.items(), key=lambda x: x[1])[0]
    info['bpm'] = int(mido.tempo2bpm(info['tempo']))

    return info


def group_channel_messages(channel_messages, channel_id):
    instrument_id = get_instrument_id(0, channel_id)
    instrument_id2messages = defaultdict(list)
    volume = default_volume

    for msg in channel_messages:
        if msg.type in messages_to_ignore:
            continue
        if msg.type not in known_messages:
            raise MidiFormatError(f"Unknown message type: {msg.type}")
        msg = copy(msg)

        if msg.type == 'program_change':
            instrument_id = get_instrument_id(msg.program, channel_id)
        elif msg.type == 'control_change' and msg.control == 7:
            volume = msg.value
        elif msg.type in ['note_on', 'note_off']:
            message_type = 'note_off' if msg.type == 'note_on' and msg.velocity == 0 else msg.type
            velocity = msg.velocity * volume / (max_velocity * max_volume)
            if message_type == 'note_on':
                assert 0 <= velocity <= 1, velocity
            instrument_id2messages[instrument_id].append(NoteMessage(
                type=message_type,
                note=msg.note,
                velocity=velocity,
                time=msg.time,
            ))

    return dict(instrument_id2messages)

# max_time=1e6
# check(msg.time <= max_time, "MIDI file too long")


def read_midi(mid):
    global_messages, channels_messages = split_channels(mid)
    info = get_midi_info(global_messages, channels_messages, mid.ticks_per_beat)
    instruments = []
    for channel_messages in channels_messages:
        channel_id = channel_messages[0].channel
        grouped_messages = group_channel_messages(channel_messages, channel_id)
        for instrument_id, messages in grouped_messages.items():
            if any(msg.type == 'note_on' for msg in messages):
                instruments.append({
                    'channel_id': channel_id,
                    'instrument_id': instrument_id,
                    'instrument_name': program2instrument[instrument_id],
                    'messages': messages,
                })

    return instruments, info


degree2accidental = {
    1.5: 'flat',
    2.5: 'flat',
    4.5: 'sharp',
    5.5: 'sharp',
    6.5: 'flat',
}


def note2scale_loc(key, octave, mode, tonic):
    tonic_interval = key2interval[tonic]
    interval = key2interval[key] - tonic_interval
    degree = mode.get_degree(interval)
    if isinstance(degree, int):
        accidental = 'none'
    else:
        relative_degree = get_relative_degree(interval, mode, major_mode)
        accidental = degree2accidental[relative_degree]
        if accidental == 'sharp':
            degree = math.floor(degree)
        elif accidental == 'flat':
            degree = math.ceil(degree)
        else:
            raise Exception("Accidental should be 'sharp' or 'flat'")
    octave = octave
    if interval < 0:
        octave -= 1
    return dict(
        octave=octave,
        degree=degree,
        accidental=accidental,
    )


def scale_loc2key_octave(octave, degree, mode, tonic, accidental=None):
    tonic_interval = key2interval[tonic]
    interval = mode.absolute_intervals[degree - 1] + tonic_interval
    if accidental == 'sharp':
        interval += 1
    elif accidental == 'flat':
        interval -= 1
    if interval < 0:
        octave -= 1
        interval += 12
    elif interval >= 12:
        octave += 1
        interval -= 12
    key = interval2key[interval]
    return key, octave


@dataclass
class Note:
    key: str = None
    octave: int = None
    time: int = None
    end_time: int = None
    note_id: int = None
    duration: int = None
    velocity: float = None
    time_sec: float = None

    scale_octave: int = None
    scale_degree: int = None
    accidental: str = 'none'

    qtime: int = None
    qduration: int = None

    bar: int = None
    beat: int = None
    beat_fraction: Fraction = None


def note_id2key_octave(note_id, pitched=True):
    if pitched:
        octave, interval = divmod(note_id, 12)
        key = interval2key[interval]
        octave = octave - 1
    else:
        key = str(note_id)
        octave = None
    return key, octave


def note2note_id(note, pitched=True):
    if pitched:
        interval = key2interval[note.key]
        return 12 * (note.octave + 1) + interval
    return int(note.key)


def get_notes_dist(info, nchannel):
    note2time = group_by(
        nchannel['notes'],
        key=lambda x: (x.key, x.octave),
        func=lambda xs: sum(x.duration * x.velocity for x in xs),
    )
    note2time = {key: mido.tick2second(time, info['ticks_per_beat'], info['tempo'])
                 for key, time in note2time.items()}
    note2time['instrument'] = nchannel['instrument_name']
    note2time['instrument_id'] = nchannel['instrument_id']
    return note2time


def get_keys_dist(info, nchannel):
    key2time = group_by(nchannel['notes'], attr='key', func=lambda xs: sum(
        x.duration * x.velocity for x in xs))
    key2time = {key: mido.tick2second(
        time, info['ticks_per_beat'], info['tempo']) for key, time in key2time.items()}
    key2time['instrument'] = nchannel['instrument_name']
    return key2time


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
        self.n_note_features = 5  # duration, velocity, sharp, flat, natural
        self.n_unpitched_features = 2  # duration, velocity

    def channel2nchannel(self, channel):
        nchannel = {k: deepcopy(v)
                    for k, v in channel.items() if k != 'messages'}
        nchannel['notes'] = []
        pitched = is_pitched(channel['instrument_id'])

        note_id2last_played_note = {}
        for msg in channel['messages']:
            if msg.type in ['note_on', 'note_off']:
                note_id = msg.note
                if note_id in note_id2last_played_note:
                    note = note_id2last_played_note[note_id]
                    note.end_time = msg.time
                    del note_id2last_played_note[note_id]
                if msg.type == 'note_on':
                    key, octave = note_id2key_octave(note_id, pitched)
                    note = Note(
                        key=key,
                        octave=octave,
                        note_id=note_id,
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
                    note_id2last_played_note[note_id] = note

        for note in nchannel['notes']:
            note.duration = note.end_time - note.time

        return nchannel

    def nchannel2kchannel(self, nchannel, in_place=False):
        kchannel = nchannel if in_place else deepcopy(nchannel)
        pitched = is_pitched(nchannel['instrument_id'])
        for note in kchannel['notes']:
            if pitched:
                scale_loc = note2scale_loc(note.key, note.octave, self.mode, self.key)
            else:
                scale_loc = dict(
                    octave=None,
                    degree=None,
                    accidental=None
                )
            note.scale_octave = scale_loc['octave']
            note.scale_degree = scale_loc['degree']
            note.accidental = scale_loc['accidental']
        return kchannel

    def kchannel2qchannel(self, kchannel, in_place=False):
        def ticks2loc(ticks):
            bar, ticks = divmod(ticks, self.info['ticks_per_bar'])
            beat, ticks = divmod(ticks, self.info['ticks_per_beat'])

            return bar, beat, ticks

        divisor2ticks = {divisor: self.info['ticks_per_beat'] / divisor
                         for divisor in self.beat_divisors}

        def note_quantizations(time):
            # todo: use numpy
            for divisor, ticks in divisor2ticks.items():
                qtime, time_error = round_number(time, ticks)
                total_error = abs(time_error)
                yield (qtime, divisor), total_error

        qchannel = kchannel if in_place else deepcopy(kchannel)

        for note in qchannel['notes']:
            time = note.time
            qtime, divisor = min(note_quantizations(time), key=lambda x: x[1])[0]
            note.qtime = int(qtime)
            note.qduration = note.end_time - note.qtime

            bar, beat, ticks = ticks2loc(note.qtime)
            quants = int(ticks // divisor2ticks[divisor])
            note.bar = int(bar)
            note.beat = int(beat)
            note.beat_fraction = Fraction(quants, divisor)

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
        for note in qchannel['notes']:
            note = copy(note)
            if pitched:
                key, octave = scale_loc2key_octave(
                    note.scale_octave,
                    note.scale_degree,
                    self.mode,
                    self.key,
                    note.accidental,
                )
                note = replace(note, key=key, octave=octave)
            note_id = note2note_id(note, pitched)
            time = loc2ticks(note.bar, note.beat, note.beat_fraction)

            note_on = NoteMessage('note_on', note=note_id, velocity=note.velocity, time=time)
            note_off = NoteMessage('note_off', note=note_id, time=time+note.qduration)
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
            duration = note.qduration / self.info['ticks_per_beat']
            features = [duration, note.velocity]
            if pitched:
                if note.accidental == 'flat':
                    v = [1., 0., 0.]
                elif note.accidental == 'none':
                    v = [0., 1., 0.]
                elif note.accidental == 'sharp':
                    v = [0., 0., 1.]
                features += v
            partial_beat[self.beat_fraction2idx[note.beat_fraction]][note_idx] = features

            bar = bars[note.bar]
            bar[note.beat] = np.maximum(bar[note.beat], partial_beat)
        vchannel = np.stack(bars)
        return vchannel

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
                        key, scale_degree, scale_octave, accidental = 4 * (None,)
                        if pitched:
                            duration, velocity, flat, natural, sharp = vnote
                            if flat:
                                accidental = 'flat'
                            elif natural:
                                accidental = 'none'
                            elif sharp:
                                accidental = 'sharp'
                            else:
                                accidental = 'none'
                            degree = note_idx % 7 + 1
                            note_idx -= degree - 1
                            octave = note_idx // 7
                            scale_degree = degree
                            scale_octave = octave
                            accidental = accidental
                        else:
                            duration, velocity = vnote
                            accidental = None
                            key = note_idx+self.min_percussion
                        note = Note(
                            key=key,
                            scale_degree=scale_degree,
                            scale_octave=scale_octave,
                            accidental=accidental,
                            bar=bar_idx,
                            beat=beat_idx,
                            beat_fraction=fraction,
                            qduration=int(duration * self.info['ticks_per_beat']),
                            velocity=velocity,
                        )
                        qchannel['notes'].append(note)
        return qchannel

    def nchannel2vchannel(self, nchannel):
        kchannel = self.nchannel2kchannel(nchannel)
        qchannel = self.kchannel2qchannel(kchannel)
        vchannel = self.qchannel2vchannel(qchannel)
        return vchannel

    def vchannel2channel(self, channel_info, vchannel):
        qchannel = self.vchannel2qchannel(channel_info, vchannel)
        channel = self.qchannel2channel(channel_info, qchannel)
        return channel

    @property
    def mode(self):
        return self.info['scale']['mode']

    @property
    def key(self):
        return self.info['scale']['key']

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
            octave = note.scale_octave
            degree = note.scale_degree
            note_idx = octave * 7 + (degree - 1)
            if note_idx < 0 or note_idx >= self.n_notes:
                raise ValueError()
            return note_idx
        note_idx = int(note.key)
        if note_idx < self.min_percussion or note_idx > self.max_percussion:
            raise ValueError()
        note_idx -= self.min_percussion
        return note_idx


def merge_channels_by_instrument(channels):
    channels_grouped = group_by(
        channels, 'instrument_name', func=merge_channels)
    channels_grouped = list(channels_grouped.values())
    return channels_grouped
