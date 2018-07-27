from copy import copy
import os
import re

from collections import defaultdict
from fractions import Fraction
from py_utils import flatten
from py_utils.math import round_number

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from style.scales import semitones2note

here = os.path.dirname(__file__)


def get_path(path):
    return os.path.join(here, path)


def parse_programs(path, group=None):
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


# def get_instrument_name(program, channel=0):
#     if channel == 9:
#         return 'Percussion'
#     return program2instrument[program]['name']


def play_midi(mid, portname=None):
    with mido.open_output(portname) as output:
        try:
            for message in mid.play():
                output.send(message)

        except KeyboardInterrupt:
            output.reset()


def create_midi(info, *channels):
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
        time, msg.time = msg.time, msg.time - time
        track.append(msg)
    track.append(MetaMessage('end_of_track', time=info['duration'] - time))

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


# todo: allow multiple instruments per channel
def split_channels(mid):
    info = {
        'ticks_per_beat': mid.ticks_per_beat,
        'bpm': [],
        'time_signature': {
            'numerator': 4,
            'denominator': 4,
            'value': 1.,
        }
    }
    channels = defaultdict(lambda: {'messages': [], 'program': 0})
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
                return None, None
            info['key'] = msg.key
        elif msg.type == 'set_tempo':
            if tempo:
                tempo2time[tempo] += msg.time - tempo_change_time
            tempo = msg.tempo
            tempo_change_time = msg.time
        elif msg.type == 'control_change':
            if msg.control == 7:
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
                    return None, None
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
        return None, None
    info['bpm'] = int(mido.tempo2bpm(info['tempo']))
    info['n_bars'] = info['duration'] / info['ticks_per_bar']

    for k, v in channels.items():
        v['index'] = k

    channels = sorted(channels.values(), key=lambda x: x['index'])
    channels = [channel for channel in channels
                if any(msg.type == 'note_on' for msg in channel['messages'])]

    for channel in channels:
        channel['instrument_id'] = get_instrument_id(channel['program'], channel['index'])
        channel['instrument_name'] = program2instrument[channel['instrument_id']]

    return channels, info


def note_number2note(n):
    octave, semitones = divmod(n, 12)
    return semitones2note[semitones], octave - 1


def channel2notes(info, channel):
    messages = channel['messages']
    channel = {k: v for k, v in channel.items() if k != 'messages'}
    channel['notes'] = []

    pitch2note = {}
    for msg in messages:
        if msg.type in ['note_on', 'note_off']:
            pitch = msg.note
            if pitch in pitch2note:
                note = pitch2note[pitch]
                note['end_time'] = msg.time
                del pitch2note[pitch]
            if msg.type == 'note_on':
                note, octave = note_number2note(pitch)
                note = dict(
                    pitch=pitch,
                    note=note,
                    octave=octave,
                    velocity=msg.velocity,
                    time=msg.time,
                    end_time=msg.time,
                    time_sec=mido.tick2second(msg.time, info['ticks_per_beat'], info['tempo'])
                )
                channel['notes'].append(note)
                pitch2note[pitch] = note

    for note in channel['notes']:
        note['duration'] = note['end_time'] - note['time']

    return channel


def quantize_notes(info, notes, beat_divisors=[8, 3]):
    def ticks2loc(ticks):
        bar, ticks = divmod(ticks, info['ticks_per_bar'])
        beat, ticks = divmod(ticks, info['ticks_per_beat'])

        return bar, beat, ticks

    divisor2ticks = {divisor: info['ticks_per_beat'] / divisor for divisor in beat_divisors}

    def note_quantizations(time, duration=None):
        # todo: use numpy
        for divisor, ticks in divisor2ticks.items():
            qtime, time_error = round_number(time, ticks)
            total_error = abs(time_error)
            yield (qtime, divisor), total_error

    for note in notes:
        time = note['time']
        duration = note.get('duration')
        qtime, divisor = min(note_quantizations(time, duration), key=lambda x: x[1])[0]
        note['qtime'] = int(qtime)
        note['qduration'] = note['end_time'] - note['qtime']

        bar, beat, ticks = ticks2loc(note['qtime'])
        note['bar'] = int(bar)
        note['beat'] = int(beat)
        quants = int(ticks // divisor2ticks[divisor])
        note['beat_fraction'] = Fraction(quants, divisor)

    return notes


def dequantize_channel(info, channel_info, quantized_channel):
    def loc2ticks(bar, beat, beat_fraction):
        return (
            bar * info['ticks_per_bar'] +
            beat * info['ticks_per_beat'] +
            int(beat_fraction * info['ticks_per_beat'])
        )

    messages = []
    channel_idx = channel_info['index']
    for note in quantized_channel:
        time = loc2ticks(note['bar'], note['beat'], note['beat_fraction'])
        note_on = Message('note_on', channel=channel_idx,
                          note=note['pitch'], velocity=note['velocity'], time=time)
        note_off = Message('note_off', channel=channel_idx,
                           note=note['pitch'], time=time+note['qduration'])
        messages += [note_on, note_off]

    channel = copy(channel_info)
    channel['messages'] = sorted(messages, key=lambda x: x.time)
    return channel


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
