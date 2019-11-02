import numpy as np
import os

import mido
import torch

from style.data import (
    included_instruments,
    get_input,
    prepare_input,
    percussion_id,
    instruments_one_hot_encoder,
    encode_instruments,
)
from style.midi import load_midi_from_file, create_midi
from style.midi_conversion import ChannelConverter, read_midi
from style.model import device, hard_output
from style.scales import major_mode, minor_mode
from style.utils.misc import make_dirs


def transfer_style(model, composition_path, style_paths, output_path):
    composition_name = os.path.splitext(os.path.basename(composition_path))[0]
    composition_input = get_model_input(composition_path)
    (_, (composition_info, composition_pitched_channels, _, composition_instruments, 
         composition_unpitched_channels)) = composition_input
    composition_cc = ChannelConverter(composition_info)
    style_, melody, rhythm = extract_style(model, composition_input)

    output_path = os.path.join(output_path, composition_name)    
    try:
        shutil.rmtree(output_path)
    except:
        pass

    save(
        composition_cc, composition_pitched_channels, composition_unpitched_channels,
        composition_instruments, os.path.join(output_path, f'original/{composition_name}.mid'))
    apply_style(model, composition_info, style_, melody, rhythm, len(composition_instruments),
                os.path.join(output_path, f'{composition_name} (reconstructed).mid'))
    for style_path in style_paths:
        style_name = os.path.splitext(os.path.basename(style_path))[0]
        style_input = get_model_input(style_path)
        _, (style_info, style_pitched_channels, _, style_instruments, style_unpitched_channels) =\
            style_input
        style_cc = ChannelConverter(style_info)
        style, _, _ = extract_style(model, style_input)

        save(style_cc, style_pitched_channels, style_unpitched_channels, style_instruments,
             os.path.join(output_path, f'original/{style_name}.mid'))
        info = combine_info(style_info=style_info, melody_info=composition_info)

        apply_style(model, info, style, melody, rhythm, len(style_instruments),
                    os.path.join(output_path, f'{composition_name} ({style_name} style).mid'))


def get_model_input(path):
    mid = load_midi_from_file(path)
    if mid is None:
        return None
    channels, info = read_midi(mid)
    channels = [c for c in channels if c['instrument_id'] in [-1, *included_instruments]]
    input = get_input(channels, info)
    return path, input


def extract_style(model, input):
    pitched_channels = input[1][1]
    max_n_bars = 1000 // pitched_channels.shape[0]
    mode, bpm, pitched_channels, instruments_features, unpitched_channels = prepare_input(
        input, max_n_bars)
    style, melody, rhythm = model.extract_style(
        mode, bpm, pitched_channels, instruments_features, unpitched_channels)
    return style.detach(), melody.detach(), rhythm.detach()


def save(cc, pitched_channels, unpitched_channels, instruments, save_path):
    channels_info = [{
        'channel_id': i,
    } for i in range(16) if i != 9][:pitched_channels.shape[1]]
    for instrument_id, channel_info in zip(instruments, channels_info):
        channel_info['instrument_id'] = instrument_id
    unpitched_channel_info = {
        'channel_id': 9,
        'instrument_id': -1,
    }

    make_dirs(os.path.dirname(save_path))

    if len(pitched_channels.shape) == 6:
        pitched_channels = torch.tensor(pitched_channels, dtype=torch.float).unsqueeze(0).to(device)
        if unpitched_channels is not None:
            unpitched_channels = torch.tensor(
                unpitched_channels, dtype=torch.float).unsqueeze(0).to(device)

    mid = decode_midi(
        cc, channels_info, pitched_channels, unpitched_channel_info, unpitched_channels)
    mid.save(save_path)


def apply_style(model, info, style, melody, rhythm, n_instruments, save_path):
    instruments_pred, mode, bpm = model.predict_song_info(style, rhythm)
    info['tempo'] = mido.bpm2tempo(round(float(bpm)))

    instruments_pred = instruments_pred.detach().cpu().numpy()
    instruments = np.argsort(-instruments_pred[0])[:n_instruments]
    if len(instruments) == 1 and instruments[0] == [percussion_id]:
        instruments = np.argsort(-instruments_pred[0])[:n_instruments+1]
    unpitched = percussion_id in instruments
    instruments = [instrument for instrument in instruments if instrument != percussion_id]

    instruments_encoded = np.zeros([len(instruments), len(included_instruments)])
    for i, instrument in enumerate(instruments):
        instruments_encoded[i, instrument] = 1
    instruments = instruments_one_hot_encoder.inverse_transform(instruments_encoded)
    instruments = [x[0] for x in instruments]

    if mode[0].argmax() == 0:
        mode = major_mode
    else:
        mode = minor_mode
    info['scale']['mode'] = mode

    cc = ChannelConverter(info)
    instruments_features = encode_instruments(instruments)
    instruments_features = torch.tensor(
        instruments_features, dtype=torch.float).to(device).unsqueeze(0)
    pitched_pred, unpitched_pred = model.apply_style(
        style, melody, rhythm, instruments_features, unpitched)

    save(cc, pitched_pred, unpitched_pred, instruments, save_path)


def combine_info(style_info, melody_info):
    info = {
        'time_signature': melody_info['time_signature'],
        'scale': style_info['scale'],
        'ticks_per_beat': melody_info['ticks_per_beat'],
        'ticks_per_bar': melody_info['ticks_per_bar'],
        'tempo': style_info['tempo'],
    }
    return info


def decode_midi(channel_converter, channels_info, pitched_channels, unpitched_channel_info=None,
                unpitched_channels=None):
    vchannels = hard_output(pitched_channels)
    vchannels = vchannels.cpu().detach().numpy()[0]
    channels = [channel_converter.vchannel2channel(channel_info, vchannel)
                for channel_info, vchannel in zip(channels_info, vchannels)]

    if unpitched_channels is not None:
        vchannels = hard_output(unpitched_channels)
        vchannel = vchannels.cpu().detach().numpy()[0, 0]
        channels.append(channel_converter.vchannel2channel(unpitched_channel_info, vchannel))

    mid = create_midi(channel_converter.info, *channels, max_delta_time=1)
    return mid
