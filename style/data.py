import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from style.exceptions import MidiFormatError
from style.midi import (
    load_midi_from_file,
    is_pitched,
    program2instrument,
    program2group,
    popular_instruments,
)
from style.midi_conversion import read_midi, ChannelConverter, get_keys_dist
from style.scales import key_names, get_scale, major_mode
from style.model import device
from style.utils.data import list2df
from style.utils.misc import group_by, flatten

included_instruments = popular_instruments
instrument_groups = [program2group[p] for p in included_instruments]
n_instruments = len(included_instruments) + 1  # also percussion

instruments_one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
instruments_one_hot_encoder.fit(np.array(included_instruments).reshape(-1, 1))

groups_one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
groups_one_hot_encoder.fit(np.array(instrument_groups).reshape(-1, 1))

instrument_size =\
    len(groups_one_hot_encoder.categories_[0]) + len(instruments_one_hot_encoder.categories_[0])
percussion_id = len(included_instruments)


def iter_all_midis(files, shuffle=False, looped=False):
    if shuffle:
        files = files[:]
        np.random.shuffle(files)
    if looped:
        while True:
            yield from iter_all_midis(files, looped=False)
    for file in files:
        mid = load_midi_from_file(file)
        if mid is not None:
            try:
                channels, info = read_midi(mid)
                yield file, channels, info
            except MidiFormatError:
                continue


def iter_inputs(files, instruments, min_n_messages=100, *args, **kwargs):
    for filename, channels, info in iter_all_midis(files, *args, **kwargs):
        channels = [
            c for c in channels
            if c['instrument_id'] in [-1, *instruments] and len(c['messages']) >= min_n_messages
        ]
        if not any(is_pitched(channel['instrument_id']) for channel in channels):
            continue
        try:
            yield filename, get_input(channels, info)
        except:
            print(filename)
            raise


def get_input(channels, info):
    cc = ChannelConverter(info)
    nchannels = [cc.channel2nchannel(channel) for channel in channels]
    instrument_id2nchannels = group_by(nchannels, 'instrument_id')
    nchannels = [merge_nchannels(nchannels) for nchannels in instrument_id2nchannels.values()]

    pitched_nchannels = [
        nchannel for nchannel in nchannels if is_pitched(nchannel['instrument_id'])
    ]
    unpitched_nchannels = [
        nchannel for nchannel in nchannels if not is_pitched(nchannel['instrument_id'])
    ]

    keys_dists = [get_keys_dist(info, nchannel) for nchannel in pitched_nchannels]
    keys_dist = keys_dists2df(keys_dists).reindex(columns=key_names).sum().fillna(0.)
    keys_dist = np.asarray(keys_dist)
    total_pitched_keys_time = keys_dist.sum()
    keys_dist /= total_pitched_keys_time

    scale = get_scale(keys_dist=keys_dist)
    info['scale'] = scale

    pitched_vchannels = [cc.nchannel2vchannel(nchannel) for nchannel in pitched_nchannels]
    unpitched_vchannels = [cc.nchannel2vchannel(nchannel) for nchannel in unpitched_nchannels]

    pitched_vchannels = np.stack(pitched_vchannels)
    if unpitched_vchannels:
        unpitched_vchannels = np.stack(unpitched_vchannels)
    else:
        unpitched_vchannels = None

    instruments = [channel['instrument_id'] for channel in pitched_nchannels]
    instruments_features = encode_instruments(instruments)

    return info, pitched_vchannels, instruments_features, instruments, unpitched_vchannels


def merge_nchannels(nchannels):
    instrument_ids = list(set(nchannel['instrument_id'] for nchannel in nchannels))
    assert len(instrument_ids) == 1
    instrument_id = instrument_ids[0]
    notes = flatten([nchannel['notes'] for nchannel in nchannels])
    notes = sorted(notes, key=lambda n: n.time)
    return {
        'channel_id': min(nchannel['channel_id'] for nchannel in nchannels),
        'instrument_id': instrument_id,
        'instrument_name': program2instrument[instrument_id],
        'notes': notes,
    }


def keys_dists2df(keys_dists):
    df = list2df(keys_dists).drop('instrument', axis=1)
    return df


def encode_instruments(instruments):
    groups = [program2group[p] for p in instruments]
    instruments = instruments_one_hot_encoder.transform(np.array(instruments).reshape(-1, 1))
    groups = groups_one_hot_encoder.transform(np.array(groups).reshape(-1, 1))
    x = np.concatenate([instruments, groups], 1)
    return x


def prepare_input(input, max_n_bars=None):
    _, (info, pitched_channels, instruments_features, _, unpitched_channels) = input

    if max_n_bars is None:
        max_n_bars = pitched_channels.shape[1]

    pitched_channels = torch.tensor(
        pitched_channels[:, :max_n_bars], dtype=torch.float).to(device).unsqueeze(0)
    instruments_features = torch.tensor(
        instruments_features, dtype=torch.float).to(device).unsqueeze(0)

    if unpitched_channels is not None:
        unpitched_channels = torch.tensor(
            unpitched_channels[:, :max_n_bars], dtype=torch.float).to(device).unsqueeze(0)

    if info['scale']['mode'] == major_mode:
        mode = torch.tensor([1., 0.])
    else:
        mode = torch.tensor([0., 1.])
    mode = mode.unsqueeze(0)
    mode = mode.to(device)

    bpm = torch.tensor(info['bpm'], dtype=torch.float)
    bpm = bpm.unsqueeze(0)
    bpm = bpm.to(device)

    return mode, bpm, pitched_channels, instruments_features, unpitched_channels


def get_used_instruments(instruments_features, unpitched_channels):
    used_instruments = instruments_features[:, :, :len(included_instruments)]
    used_instruments = used_instruments.sum(1)
    used_instruments = (used_instruments > 0).float()

    percussion_used = unpitched_channels is not None
    percussion_used = torch.tensor(percussion_used, dtype=torch.float).to(device)
    percussion_used = percussion_used.view(1, 1)

    used_instruments = torch.cat([used_instruments, percussion_used], 1)
    return used_instruments
