import copy
import os
import re

from collections import defaultdict
from py_utils import flatten

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack


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
                program2instrument[int(program)] = {
                    'group': group,
                    'name': name,
                }
            else:
                group = line
    return program2instrument


program2instrument = parse_programs(get_path('midi_programs.txt'))


def get_instrument_name(program, channel=0):
    if channel == 9:
        return 'Percussion'
    if program == 0:
        program = 1
    return program2instrument[program]['name']


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
        track.append(Message('program_change',
                             channel=channel['index'], program=channel['program']))

    msgs = flatten([channel['messages'] for channel in channels])
    msgs = sorted(msgs, key=lambda x: x.time)

    time = 0
    for msg in msgs:
        msg = copy.copy(msg)
        time, msg.time = msg.time, msg.time - time
        track.append(msg)
    track.append(MetaMessage('end_of_track', time=info['duration'] - time))

    return mid


def merge_tracks(tracks):
    msgs = []
    for track in tracks:
        time = 0
        for msg in track:
            msg = copy.copy(msg)
            time += msg.time
            msg.time = time
            msgs.append(msg)
    msgs = sorted(msgs, key=lambda x: x.time)
    return msgs


def split_channels(mid):
    info = {
        'ticks_per_beat': mid.ticks_per_beat,
        'bpm': [],
    }
    channels = defaultdict(lambda: {'messages': []})
    played_channels = set()

    messages_to_ignore = [
        'smpte_offset',
        'midi_port',
        'sysex',
        'end_of_track',
        'track_name',
        'copyright',
        'lyrics',
    ]

    tempo = None
    tempo_change_time = 0
    tempo2time = defaultdict(int)

    for msg in merge_tracks(mid.tracks):
        msg = copy.copy(msg)
        if msg.type in messages_to_ignore:
            continue

        if msg.type == 'time_signature':
            assert not played_channels
            info['time_signature'] = {
                'numerator': msg.numerator,
                'denominator': msg.denominator,
                'value': msg.numerator / msg.denominator
            }
        elif msg.type == 'key_signature':
            assert not played_channels
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
            assert msg.channel not in played_channels
            channels[msg.channel]['program'] = msg.program
        elif msg.type in ['note_on', 'note_off', 'pitchwheel']:
            if msg.type == 'note_on' and msg.velocity == 0:
                msg = Message('note_off', channel=msg.channel, note=msg.note,
                              velocity=msg.velocity, time=msg.time)
            channels[msg.channel]['messages'].append(msg)
            if msg.type in ['note_on', 'note_off']:
                played_channels.add(msg.channel)
        else:
            raise Exception(f'Unknown message type: {msg.type}')

    tempo2time[tempo] += msg.time - tempo_change_time
    info['duration'] = msg.time
    info['ticks_per_bar'] = int(mid.ticks_per_beat * 4 * info['time_signature']['value'])

    tempo2time = {k: v for k, v in tempo2time.items() if v}
    info['tempo2time'] = tempo2time
    info['tempo'] = max(tempo2time.items(), key=lambda x: x[1])[0]
    info['bpm'] = int(mido.tempo2bpm(info['tempo']))
    info['n_bars'] = info['duration'] / info['ticks_per_bar']

    for k, v in channels.items():
        v['index'] = k

    channels = sorted(channels.values(), key=lambda x: x['index'])
    channels = [channel for channel in channels
                if any(msg.type == 'note_on' for msg in channel['messages'])]

    for channel in channels:
        channel['instrument_name'] = get_instrument_name(channel['program']+1, channel['index'])

    return channels, info
