from copy import copy
import math
import os
import re

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack, KeySignatureError


here = os.path.dirname(__file__)


def get_path(path):
    return os.path.join(here, path)


default_tempo = 500000
default_volume = 96
max_volume = 127
max_velocity = 127


def parse_programs(path):
    program2instrument = {}
    program2group = {}
    group_name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'([0-9]+) (.*)', line)
            if m:
                program, name = m.groups()
                program2instrument[int(program)-1] = name
                program2group[int(program) - 1] = group_name
            else:
                group_name = line
    return program2instrument, program2group


program2instrument, program2group = parse_programs(get_path('midi_programs.txt'))

program2instrument[-1] = 'Percussion'
# program2group[-1] = 'Percussion'


def get_instrument_id(program, channel=0):
    if channel == 9:
        return -1
    return program


def is_sound_effect(instrument_id):
    return instrument_id > 119


def is_pitched(instrument_id):
    return instrument_id >= 0 and not is_sound_effect(instrument_id)


def load_midi_from_file(path):
    try:
        return MidiFile(path)
    except (OSError, ValueError, KeyError, EOFError, KeySignatureError):
        return None


def play_midi(mid, portname=None):
    with mido.open_output(portname) as output:
        try:
            for message in mid.play():
                output.send(message)
        except KeyboardInterrupt:
            output.reset()


# todo: support multiple instruments in single channel
def create_midi(info, *instruments, max_delta_time=math.inf):
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

    messages = []
    for instrument in instruments:
        if instrument['channel_id'] != 9:
            track.append(Message(
                'program_change',
                channel=instrument['channel_id'],
                program=instrument['instrument_id'],
                time=0,
            ))
        for msg in instrument['messages']:
            velocity = int(msg.velocity * max_velocity)
            assert velocity <= 127, (velocity, msg.velocity)
            messages.append(Message(
                msg.type,
                channel=instrument['channel_id'],
                note=msg.note,
                velocity=velocity,
                time=msg.time,
            ))

    messages = sorted(messages, key=lambda x: x.time)
    messages.append(MetaMessage('end_of_track', time=info['duration']))

    time = 0
    for msg in messages:
        msg = copy(msg)
        delta_time = min(msg.time - time, max_delta_time)
        time = msg.time
        msg.time = max(0, delta_time)
        track.append(msg)
    return mid
