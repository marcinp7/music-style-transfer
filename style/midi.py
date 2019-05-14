from copy import copy
import math
import os
import re

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from py_utils import flatten


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
    msgs.append(MetaMessage('end_of_track', time=info['duration']))

    time = 0
    for msg in msgs:
        msg = copy(msg)
        delta_time = min(msg.time - time, max_delta_time)
        time = msg.time
        msg.time = delta_time
        track.append(msg)
    return mid
