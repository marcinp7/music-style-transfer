import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from style.utils.pytorch import Distributed, squash_dims, LSTM, cat_with_broadcast

epsilon = 1e-7

n_beat_fractions = 10
n_pitched_features = 5
n_unpitched_features = 2
n_octaves = 8
n_scale_degrees = 7
n_pitched_notes = n_octaves * n_scale_degrees
n_unpitched_notes = 47
n_modes = 2

min_bpm = 50
max_bpm = 200

bpm_range = max_bpm - min_bpm


def get_mean_size(*values, factor=1):
    mean = np.mean(values) * factor
    return math.ceil(mean)


class PitchedChannelsEncoder(nn.Module):
    def __init__(self, beat_size, bar_size, instrument_size):
        super().__init__()
        assert bar_size % 2 == 0

        beats_conv_in_channels = n_beat_fractions * n_pitched_features
        beats_conv_out_channels = get_mean_size(beats_conv_in_channels, beat_size)
        instruments_linear_size = get_mean_size(instrument_size, beat_size)
        linear_size = beat_size

        self.beats_conv = nn.Conv1d(
            in_channels=beats_conv_in_channels,
            out_channels=beats_conv_out_channels,
            kernel_size=2*n_scale_degrees,
            stride=n_scale_degrees,
            padding=4,
        )
        self.beats_conv = Distributed(self.beats_conv, depth=3)
        self.instruments_linear = nn.Linear(
            in_features=instrument_size,
            out_features=instruments_linear_size,
        )
        self.linear = nn.Linear(
            in_features=beats_conv_out_channels*n_octaves+instruments_linear_size,
            out_features=linear_size,
        )
        self.beats_lstm = LSTM(
            input_size=linear_size,
            hidden_size=beat_size,
            num_layers=1,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=2)
        self.bars_lstm = LSTM(
            input_size=beat_size,
            hidden_size=bar_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, channels, instruments_features):
        x = channels.transpose(-1, -2)
        # (batch, channel, bar, beat, beat_fraction, note_features, note)
        x = x.contiguous()
        x = squash_dims(x, 4, 6)  # (batch, channel, bar, beat, features, note)
        x = self.beats_conv(x)  # (batch, channel, bar, beat, features, octave)
        x = F.leaky_relu(x)
        x1 = squash_dims(x, -2)  # (batch, channel, bar, beat, features)

        x = self.instruments_linear(instruments_features)  # (batch, channel, features)
        x = F.leaky_relu(x)
        x2 = x.view(*x.shape[:2], 1, 1, -1)  # (batch, channel, bar, beat, features)

        x = cat_with_broadcast([x1, x2], -1)  # (batch, channel, bar, beat, features)
        x = self.linear(x)
        x = F.leaky_relu(x)
        beats = self.beats_lstm(x)[0]

        x = beats[:, :, :, -1]  # (batch, channel, bar, features)
        x = combine(x, dim=1)  # (batch, bar, features)
        bars = self.bars_lstm(x)[0]  # (batch, bar, features)

        return beats, bars


class UnpitchedChannelsEncoder(nn.Module):
    def __init__(self, beat_size, bar_size):
        super().__init__()
        assert bar_size % 2 == 0

        linear_size = beat_size

        self.linear = nn.Linear(
            in_features=n_beat_fractions*n_unpitched_notes*n_unpitched_features,
            out_features=linear_size,
        )
        self.beats_lstm = LSTM(
            input_size=linear_size,
            hidden_size=beat_size,
            num_layers=1,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=2)
        self.bars_lstm = LSTM(
            input_size=beat_size,
            hidden_size=bar_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, channels):
        x = channels.transpose(-1, -2)
        # (batch, channel, bar, beat, beat_fraction, note_features, note)
        x = x.contiguous()
        x = squash_dims(x, 4, 7)  # (batch, channel, bar, beat, features)
        x = self.linear(x)
        x = F.leaky_relu(x)
        beats = self.beats_lstm(x)[0]  # (batch, channel, bar, beat, features)

        x = beats[:, :, :, -1]  # (batch, channel, bar, features)
        x = combine(x, dim=1)  # (batch, bar, features)
        bars = self.bars_lstm(x)[0]  # (batch, bar, features)

        return beats, bars


class StyleEncoder(nn.Module):
    def __init__(self, style_size, bar_size, instrument_size):
        super().__init__()

        bars_lstm_size = get_mean_size(bar_size, style_size)
        instruments_linear_size = get_mean_size(instrument_size, style_size, factor=.25)
        mode_linear_size = get_mean_size(n_modes, style_size, factor=.1)
        bpm_linear_size = get_mean_size(style_size, 1, factor=.05)

        linear_input_size = sum([
            bars_lstm_size, instruments_linear_size, mode_linear_size, bpm_linear_size])

        self.bars_lstm = LSTM(
            input_size=bar_size,
            hidden_size=bars_lstm_size,
            num_layers=1,
            batch_first=True,
        )
        self.instruments_linear = nn.Linear(
            in_features=instrument_size,
            out_features=instruments_linear_size,
        )
        self.mode_linear = nn.Linear(
            in_features=n_modes,
            out_features=mode_linear_size,
        )
        self.bpm_linear = nn.Linear(
            in_features=1,
            out_features=bpm_linear_size,
        )
        self.linear = nn.Linear(
            in_features=linear_input_size,
            out_features=style_size,
        )

    def forward(self, bars, instruments_features, mode, bpm):
        x = self.bars_lstm(bars)[0]  # (batch, bar, features)
        x = x[:, -1]  # (batch, features)
        x1 = x.unsqueeze(1)  # (batch, channel, features)

        x = self.instruments_linear(instruments_features)  # (batch, channel, features)
        x2 = F.leaky_relu(x)

        x = self.mode_linear(mode)  # (batch, features)
        x = F.leaky_relu(x)
        x3 = x.unsqueeze(1)  # (batch, channel, features)

        x = bpm.unsqueeze(-1)  # (batch, features)
        x = self.bpm_linear(x)
        x = F.leaky_relu(x)
        x4 = x.unsqueeze(1)  # (batch, channel, features)

        x = cat_with_broadcast([x1, x2, x3, x4], -1)  # (batch, channel, features)
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = combine(x, dim=1)  # (batch, features)
        return x


class MelodyEncoder(nn.Module):
    def __init__(self, melody_size, beat_size, bar_size, instrument_size):
        super().__init__()

        beats_linear_size = get_mean_size(beat_size, melody_size)
        bars_linear_size = get_mean_size(bar_size, melody_size)
        instrument_linear_size = get_mean_size(instrument_size, melody_size, factor=.25)
        linears_size = melody_size
        channels_linear_size = get_mean_size(n_pitched_features, melody_size)

        linears_input_size = sum([beats_linear_size, bars_linear_size, instrument_linear_size])

        # self.beat_conv = nn.Conv1d(
        #     in_channels=5*10,
        #     out_channels=7,
        #     kernel_size=14,
        #     stride=7,
        #     padding=4,
        # )
        # self.beat_conv = Distributed(self.beat_conv, depth=3)
        self.beats_linear = nn.Linear(
            in_features=beat_size,
            out_features=beats_linear_size,
        )
        self.bars_linear = nn.Linear(
            in_features=bar_size,
            out_features=bars_linear_size,
        )
        self.instruments_linear = nn.Linear(
            in_features=instrument_size,
            out_features=instrument_linear_size,
        )
        self.octave_linear = nn.Linear(
            in_features=linears_input_size,
            out_features=linears_size*n_octaves,
        )
        self.scale_degree_linear = nn.Linear(
            in_features=linears_input_size,
            out_features=linears_size*n_scale_degrees,
        )
        self.channels_linear = nn.Linear(
            in_features=n_pitched_features,
            out_features=channels_linear_size,
        )
        self.linear = nn.Linear(
            in_features=linears_size+channels_linear_size,
            out_features=melody_size,
        )

    def forward(self, beats, bars, channels, instruments):
        x = self.beats_linear(beats)  # (batch, channel, bar, beat, features)
        x = F.leaky_relu(x)
        x1 = x.unsqueeze(-2)  # (batch, channel, bar, beat, beat_fraction, features)

        x = self.bars_linear(bars)  # (batch, bar, features)
        x = F.leaky_relu(x)
        x2 = x.view(x.shape[0], 1, x.shape[1], 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = self.instruments_linear(instruments)  # (batch, channel, features)
        x = F.leaky_relu(x)
        x3 = x.view(*x.shape[:2], 1, 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        y = cat_with_broadcast([x1, x2, x3], -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = self.octave_linear(y)  # (batch, channel, bar, beat, beat_fraction, features)
        x = x.view(*x.shape[:-1], n_octaves, -1)
        # (batch, channel, bar, beat, beat_fraction, octave, features)
        x = F.leaky_relu(x)
        x1 = x.unsqueeze(-2)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x = self.scale_degree_linear(y)  # (batch, channel, bar, beat, beat_fraction, features)
        x = x.view(*x.shape[:-1], n_scale_degrees, -1)
        # (batch, channel, bar, beat, beat_fraction, scale_degree, features)
        x = F.leaky_relu(x)
        x2 = x.unsqueeze(-3)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x = x1 + x2  # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)
        x = F.leaky_relu(x)
        x1 = squash_dims(x, 5, 7)  # (batch, channel, bar, beat, beat_fraction, note, features)

        x = self.channels_linear(channels)
        # (batch, channel, bar, beat, beat_fraction, note, features)
        x2 = F.leaky_relu(x)

        x = cat_with_broadcast([x1, x2], -1)
        # (batch, channel, bar, beat, beat_fraction, note, features)
        x = self.linear(x)  # (batch, channel, bar, beat, beat_fraction, note, features)
        x = F.leaky_relu(x)
        x = combine(x, dim=1)  # (batch, bar, beat, beat_fraction, note, features)
        return x


# todo: channels_linear
class PitchedRhythmEncoder(nn.Module):
    def __init__(self, rhythm_size, beat_size, bar_size, instrument_size):
        super().__init__()

        beats_linear_size = get_mean_size(beat_size, rhythm_size)
        bars_linear_size = get_mean_size(bar_size, rhythm_size, factor=.5)
        channels_linear_size = get_mean_size(n_pitched_notes * n_pitched_features, rhythm_size,
                                             factor=.1)
        instruments_linear_size = get_mean_size(instrument_size, rhythm_size, factor=.5)
        mode_linear_size = get_mean_size(n_modes, rhythm_size, factor=.25)
        bpm_linear_size = get_mean_size(1, rhythm_size, factor=.25)

        linear_input_size = sum([
            beats_linear_size, bars_linear_size, channels_linear_size, instruments_linear_size,
            mode_linear_size, bpm_linear_size])

        self.beats_linear = nn.Linear(
            in_features=beat_size,
            out_features=beats_linear_size,
        )
        self.bars_linear = nn.Linear(
            in_features=bar_size,
            out_features=bars_linear_size,
        )
        self.channels_linear = nn.Linear(
            in_features=n_pitched_notes*n_pitched_features,
            out_features=channels_linear_size,
        )
        self.instruments_linear = nn.Linear(
            in_features=instrument_size,
            out_features=instruments_linear_size,
        )
        self.mode_linear = nn.Linear(
            in_features=n_modes,
            out_features=mode_linear_size,
        )
        self.bpm_linear = nn.Linear(
            in_features=1,
            out_features=bpm_linear_size,
        )
        self.linear = nn.Linear(
            in_features=linear_input_size,
            out_features=rhythm_size,
        )

    def forward(self, beats, bars, channels, instruments_features, mode, bpm):
        x = self.beats_linear(beats)  # (batch, channel, bar, beat, features)
        x = F.leaky_relu(x)
        x1 = x.unsqueeze(4)  # (batch, channel, bar, beat, beat_fraction, features)

        x = self.bars_linear(bars)  # (batch, bar, features)
        x = F.leaky_relu(x)
        x2 = x.view(x.shape[0], 1, x.shape[1], 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = squash_dims(channels, -2)  # (batch, channel, bar, beat, beat_fraction, features)
        x = self.channels_linear(x)
        x3 = F.leaky_relu(x)

        x = self.instruments_linear(instruments_features)  # (batch, channel, features)
        x = F.leaky_relu(x)
        x4 = x.view(*x.shape[:2], 1, 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = self.mode_linear(mode)  # (batch, features)
        x = F.leaky_relu(x)
        x5 = x.view(x.shape[0], 1, 1, 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = bpm.unsqueeze(-1)  # (batch, features)
        x = self.bpm_linear(x)
        x = F.leaky_relu(x)
        x6 = x.view(x.shape[0], 1, 1, 1, 1, x.shape[1])
        # (batch, channel, bar, beat, beat_fraction, features)

        x = cat_with_broadcast([x1, x2, x3, x4, x5, x6], -1)
        # (batch, channel, bar, beat, beat_fraction, features)
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = combine(x, dim=1)  # (batch, bar, beat, beat_fraction, features)
        return x


class UnpitchedRhythmEncoder(nn.Module):
    def __init__(self, rhythm_size, beat_size, bar_size):
        super().__init__()

        beats_linear_size = get_mean_size(beat_size, rhythm_size)
        bars_linear_size = get_mean_size(bar_size, rhythm_size, factor=.5)
        channels_linear_size = get_mean_size(n_unpitched_notes * n_unpitched_features, rhythm_size,
                                             factor=.25)
        bpm_linear_size = get_mean_size(1, rhythm_size, factor=.25)

        linear_input_size = sum([
            beats_linear_size, bars_linear_size, channels_linear_size, bpm_linear_size])

        self.beats_linear = nn.Linear(
            in_features=beat_size,
            out_features=beats_linear_size,
        )
        self.bars_linear = nn.Linear(
            in_features=bar_size,
            out_features=bars_linear_size,
        )
        self.channels_linear = nn.Linear(
            in_features=n_unpitched_notes*n_unpitched_features,
            out_features=channels_linear_size,
        )
        self.bpm_linear = nn.Linear(
            in_features=1,
            out_features=bpm_linear_size,
        )
        self.linear = nn.Linear(
            in_features=linear_input_size,
            out_features=rhythm_size,
        )

    def forward(self, beats, bars, channels, bpm):
        x = self.beats_linear(beats)  # (batch, channel, bar, beat, features)
        x = F.leaky_relu(x)
        x1 = x.unsqueeze(4)  # (batch, channel, bar, beat, beat_fraction, features)

        x = self.bars_linear(bars)  # (batch, bar, features)
        x = F.leaky_relu(x)
        x2 = x.view(x.shape[0], 1, x.shape[1], 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = squash_dims(channels, -2)  # (batch, channel, bar, beat, beat_fraction, features)
        x = self.channels_linear(x)
        x3 = F.leaky_relu(x)

        x = bpm.unsqueeze(-1)  # (batch, features)
        x = self.bpm_linear(x)
        x = F.leaky_relu(x)
        x4 = x.view(x.shape[0], 1, 1, 1, 1, x.shape[1])
        # (batch, channel, bar, beat, beat_fraction, features)

        x = cat_with_broadcast([x1, x2, x3, x4], -1)
        # (batch, channel, bar, beat, beat_fraction, features)
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = combine(x, dim=1)  # (batch, bar, beat, beat_fraction, features)
        return x


class SongInfoModel(nn.Module):
    def __init__(self, n_rhythm_features, style_size, rhythm_size, n_instruments):
        super().__init__()

        beats_lstm_size = get_mean_size(n_beat_fractions * rhythm_size, n_rhythm_features,
                                        factor=.05)

        style_instruments_linear_size = get_mean_size(style_size, n_instruments, factor=.05)
        rhythm_instruments_linear_size = get_mean_size(rhythm_size, n_instruments, factor=.25)

        style_mode_linear_size = get_mean_size(style_size, n_modes, factor=.01)
        rhythm_mode_linear_size = get_mean_size(rhythm_size, n_modes, factor=.1)

        style_bpm_linear_size = get_mean_size(style_size, 1, factor=.01)
        rhythm_bpm_linear_size = get_mean_size(rhythm_size, 1, factor=.1)

        self.beats_lstm = LSTM(
            input_size=n_beat_fractions*rhythm_size,
            hidden_size=beats_lstm_size,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=1)
        self.bars_lstm = LSTM(
            input_size=beats_lstm_size,
            hidden_size=n_rhythm_features,
            batch_first=True,
        )

        self.style_instruments_linear = nn.Linear(
            in_features=style_size,
            out_features=style_instruments_linear_size,
        )
        self.rhythm_instruments_linear = nn.Linear(
            in_features=n_rhythm_features,
            out_features=rhythm_instruments_linear_size,
        )
        self.instruments_linear = nn.Linear(
            in_features=style_instruments_linear_size+rhythm_instruments_linear_size,
            out_features=n_instruments,
        )

        self.style_mode_linear = nn.Linear(
            in_features=style_size,
            out_features=style_mode_linear_size,
        )
        self.rhythm_mode_linear = nn.Linear(
            in_features=n_rhythm_features,
            out_features=rhythm_mode_linear_size,
        )
        self.mode_linear = nn.Linear(
            in_features=style_mode_linear_size+rhythm_mode_linear_size,
            out_features=n_modes,
        )

        self.style_bpm_linear = nn.Linear(
            in_features=style_size,
            out_features=style_bpm_linear_size,
        )
        self.rhythm_bpm_linear = nn.Linear(
            in_features=n_rhythm_features,
            out_features=rhythm_bpm_linear_size,
        )
        self.bpm_linear = nn.Linear(
            in_features=style_bpm_linear_size+rhythm_bpm_linear_size,
            out_features=1,
        )

    def get_rhythm_features(self, rhythm):
        x = squash_dims(rhythm, -2)  # (batch, bar, beat, features)
        x = self.beats_lstm(x)[0]  # (batch, bar, beat, features)
        x = x[:, :, -1]  # (batch, bar, features)
        x = self.bars_lstm(x)[0]  # (batch, bar, features)
        x = x[:, -1]  # (batch, features)
        return x

    def predict_instruments(self, style, rhythm_features):
        x = self.style_instruments_linear(style)  # (batch, features)
        x1 = F.leaky_relu(x)

        x = self.rhythm_instruments_linear(rhythm_features)  # (batch, features)
        x2 = F.leaky_relu(x)

        x = cat_with_broadcast([x1, x2], -1)  # (batch, features)
        x = self.instruments_linear(x)  # (batch, features)
        return x

    def predict_mode(self, style, rhythm_features):
        x = self.style_mode_linear(style)  # (batch, features)
        x1 = F.leaky_relu(x)

        x = self.rhythm_mode_linear(rhythm_features)  # (batch, features)
        x2 = F.leaky_relu(x)

        x = cat_with_broadcast([x1, x2], -1)  # (batch, features)
        x = self.mode_linear(x)
        return x

    def predict_bpm(self, style, rhythm_features):
        x = self.style_bpm_linear(style)  # (batch, features)
        x1 = F.leaky_relu(x)

        x = self.rhythm_bpm_linear(rhythm_features)  # (batch, features)
        x2 = F.leaky_relu(x)

        x = cat_with_broadcast([x1, x2], -1)  # (batch, features)
        x = self.bpm_linear(x)  # (batch, features)
        x = x[:, 0]  # (batch,)
        x = torch.sigmoid(x)
        x = x * bpm_range + min_bpm
        return x

    def forward(self, style, rhythm):
        rhythm_features = self.get_rhythm_features(rhythm)
        instruments = self.predict_instruments(style, rhythm_features)
        mode = self.predict_mode(style, rhythm_features)
        bpm = self.predict_bpm(style, rhythm_features)
        return instruments, mode, bpm


def duration_activation(x, max_duration=6):
    x = torch.sigmoid(x) * max_duration
    return x


def velocity_activation(x):
    # x = torch.tanh(x)
    # x = torch.relu(x)
    x = torch.sigmoid(x)
    return x


def accidentals_activation(x):
    x = torch.sigmoid(x)
    return x


class PitchedStyleApplier(nn.Module):
    def __init__(self, style_size, melody_size, rhythm_size, instrument_size):
        super().__init__()

        style_linear_size = get_mean_size(style_size, n_pitched_features, factor=.5)
        rhythm_linear_size = get_mean_size(rhythm_size, n_pitched_features, factor=.5)
        instruments_linear_size = get_mean_size(instrument_size, n_pitched_features, factor=.4)
        linears_output_size = n_pitched_features * 6
        melody_linear_size = get_mean_size(melody_size, n_pitched_features, factor=3)

        linear_input_size = sum([
            style_linear_size, rhythm_linear_size, instruments_linear_size])

        self.style_linear = nn.Linear(
            in_features=style_size,
            out_features=style_linear_size,
        )
        self.rhythm_linear = nn.Linear(
            in_features=rhythm_size,
            out_features=rhythm_linear_size,
        )
        self.instruments_linear = nn.Linear(
            in_features=instrument_size,
            out_features=instruments_linear_size,
        )
        self.octave_linear = nn.Linear(
            in_features=linear_input_size,
            out_features=linears_output_size*n_octaves,
        )
        self.scale_degree_linear = nn.Linear(
            in_features=linear_input_size,
            out_features=linears_output_size*n_scale_degrees,
        )
        self.melody_linear = nn.Linear(
            in_features=melody_size,
            out_features=melody_linear_size,
        )
        self.linear = nn.Linear(
            in_features=linears_output_size+melody_linear_size,
            out_features=n_pitched_features,
        )

    def forward(self, style, melody, rhythm, instruments):
        x = self.style_linear(style)  # (batch, features)
        x = F.leaky_relu(x)
        x1 = x.view(x.shape[0], 1, 1, 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = self.rhythm_linear(rhythm)
        # (batch, channel, bar, beat, beat_fraction, features)
        x = F.leaky_relu(x)
        x2 = x.view(x.shape[0], 1, *x.shape[1:4], -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = self.instruments_linear(instruments)  # (batch, channel, features)
        x = F.leaky_relu(x)
        x3 = x.view(*x.shape[:2], 1, 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        y = cat_with_broadcast([x1, x2, x3], -1)
        # (batch, channel, bar, beat, beat_fraction, features)

        x = self.octave_linear(y)  # (batch, channel, bar, beat, beat_fraction, features)
        x = x.view(*x.shape[:-1], n_octaves, -1)
        # (batch, channel, bar, beat, beat_fraction, octave, features)
        x = F.leaky_relu(x)
        x1 = x.unsqueeze(-2)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x = self.scale_degree_linear(y)  # (batch, channel, bar, beat, beat_fraction, features)
        x = x.view(*x.shape[:-1], n_scale_degrees, -1)
        # (batch, channel, bar, beat, beat_fraction, scale_degree, features)
        x = F.leaky_relu(x)
        x2 = x.unsqueeze(-3)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x = x1 + x2  # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)
        x = F.leaky_relu(x)
        x1 = squash_dims(x, 5, 7)  # (batch, channel, bar, beat, beat_fraction, note, features)

        x = self.melody_linear(melody)
        x = F.leaky_relu(x)
        x2 = x.unsqueeze(1)  # (batch, channel, bar, beat, beat_fraction, note, features)

        x = cat_with_broadcast([x1, x2], -1)
        # (batch, channel, bar, beat, beat_fraction, note, features)
        x = self.linear(x)  # (batch, channel, bar, beat, beat_fraction, note, note_features)

        duration = duration_activation(x[:, :, :, :, :, :, :1])
        velocity = velocity_activation(x[:, :, :, :, :, :, 1:2])
        accidentals = accidentals_activation(x[:, :, :, :, :, :, 2:])
        x = torch.cat([duration, velocity, accidentals], -1)
        # (batch, channel, bar, beat, beat_fraction, note, note_features)
        return x


class UnpitchedStyleApplier(nn.Module):
    def __init__(self, style_size, rhythm_size):
        super().__init__()

        style_linear_size = get_mean_size(style_size, n_unpitched_features, factor=.5)
        rhythm_linear_size = get_mean_size(rhythm_size, n_unpitched_features, factor=1)
        notes_linear_size = n_unpitched_features * 4

        self.style_linear = nn.Linear(
            in_features=style_size,
            out_features=n_beat_fractions*style_linear_size,
        )
        self.rhythm_linear = nn.Linear(
            in_features=rhythm_size,
            out_features=rhythm_linear_size,
        )
        self.notes_linear = nn.Linear(
            in_features=style_linear_size+rhythm_linear_size,
            out_features=n_unpitched_notes*notes_linear_size,
        )
        self.linear = nn.Linear(
            in_features=notes_linear_size,
            out_features=n_unpitched_features,
        )

    def forward(self, style, rhythm):
        x = self.style_linear(style)  # (batch, features)
        x = F.leaky_relu(x)
        x1 = x.view(x.shape[0], 1, 1, n_beat_fractions, -1)
        # (batch, bar, beat, beat_fraction, features)

        x = self.rhythm_linear(rhythm)  # (batch, bar, beat, beat_fraction, features)
        x2 = F.leaky_relu(x)

        x = cat_with_broadcast([x1, x2], -1)  # (batch, bar, beat, beat_fraction, features)
        x = self.notes_linear(x)
        x = F.leaky_relu(x)
        x = x.view(*x.shape[:4], n_unpitched_notes, -1)
        # (batch, bar, beat, beat_fraction, note, features)
        x = self.linear(x)  # (batch, bar, beat, beat_fraction, note, note_features)

        duration = duration_activation(x[:, :, :, :, :, :1])
        velocity = velocity_activation(x[:, :, :, :, :, 1:2])
        x = torch.cat([duration, velocity], 5)
        # (batch, bar, beat, beat_fraction, note, note_features)
        x = x.unsqueeze(1)  # (batch, channel, bar, beat, beat_fraction, note, note_features)
        return x


class StyleTransferModel(nn.Module):
    def __init__(
            self,
            pitched_channels_encoder, unpitched_channels_encoder,
            style_encoder, melody_encoder,
            pitched_rhythm_encoder, unpitched_rhythm_encoder,
            song_info_model,
            pitched_style_applier, unpitched_style_applier,
        ):
        super().__init__()
        self.pitched_channels_encoder = pitched_channels_encoder
        self.unpitched_channels_encoder = unpitched_channels_encoder

        self.style_encoder = style_encoder
        self.melody_encoder = melody_encoder

        self.pitched_rhythm_encoder = pitched_rhythm_encoder
        self.unpitched_rhythm_encoder = unpitched_rhythm_encoder

        self.song_info_model = song_info_model

        self.pitched_style_applier = pitched_style_applier
        self.unpitched_style_applier = unpitched_style_applier

    def extract_style(self, mode, bpm, pitched_channels, instruments_features,
                      unpitched_channels=None):
        pitched_beats, pitched_bars = self.pitched_channels_encoder(
            pitched_channels, instruments_features)
        pitched_rhythm = self.pitched_rhythm_encoder(
            pitched_beats, pitched_bars, pitched_channels, instruments_features, mode, bpm)

        if unpitched_channels is None:
            bars = pitched_bars
            rhythm = pitched_rhythm
        else:
            unpitched_beats, unpitched_bars = self.unpitched_channels_encoder(unpitched_channels)
            unpitched_rhythm = self.unpitched_rhythm_encoder(
                unpitched_beats, unpitched_bars, unpitched_channels, bpm)

            bars = combine(pitched_bars, unpitched_bars)
            rhythm = combine(pitched_rhythm, unpitched_rhythm)

        style = self.style_encoder(bars, instruments_features, mode, bpm)
        melody = self.melody_encoder(
            pitched_beats, pitched_bars, pitched_channels, instruments_features)

        return style, melody, rhythm

    def predict_song_info(self, style, rhythm):
        instruments, mode, bpm = self.song_info_model(style, rhythm)
        return instruments, mode, bpm

    def apply_style(self, style, melody, rhythm, instruments_features, unpitched=False):
        x_pitched = self.pitched_style_applier(style, melody, rhythm, instruments_features)
        x_unpitched = self.unpitched_style_applier(style, rhythm) if unpitched else None
        return x_pitched, x_unpitched

    def forward(self, mode, bpm, pitched_channels, instruments_features, unpitched_channels=None):
        # channels: (batch, channel, bar, beat, beat_fraction, note, note_features)
        # instruments_features: (batch, channel, features)

        style, melody, rhythm = self.extract_style(
            mode, bpm, pitched_channels, instruments_features, unpitched_channels)
        instruments_pred, mode_pred, bpm_pred = self.predict_song_info(style, rhythm)
        x_pitched, x_unpitched = self.apply_style(
            style, melody, rhythm, instruments_features, unpitched_channels is not None)
        return (instruments_pred, mode_pred, bpm_pred), x_pitched, x_unpitched


def combine(*tensors, dim=None, safe=True):
    assert len(tensors)
    if len(tensors) == 1:
        tensor = tensors[0]
    else:
        tensor = torch.stack(tensors)
        dim = 0
    if dim is None:
        return tensor

    x = tensor ** 2
    dims = [i for i in range(len(tensor.shape)) if i != dim]
    x = x.sum(dims, keepdim=True)
    if safe:
        norm = torch.sqrt(1. + x)
    else:
        norm = torch.sqrt(x)

    x = tensor * norm
    return x.sum(dim) / norm.sum()


def hard_output(x):
    duration = x[:, :, :, :, :, :, :1]
    velocity = x[:, :, :, :, :, :, 1:2]

    velocity *= (velocity > .01).float()

    if x.shape[-1] > 2:
        accidentals = x[:, :, :, :, :, :, 2:]
        max_accidentals = accidentals.max(dim=-1)[0]
        new_accidentals = accidentals == max_accidentals.unsqueeze(-1)
        new_accidentals *= accidentals > .1
        x = torch.cat([duration, velocity, new_accidentals.float()], -1)
    else:
        x = torch.cat([duration, velocity], -1)
    return x


def get_duration(x):
    return x[:, :, :, :, :, :, 0]


def get_velocity(x):
    return x[:, :, :, :, :, :, 1]


def get_accidentals(x):
    return x[:, :, :, :, :, :, 2:]


def get_duration_loss(input, target, mask):
    x = ((input - target.clamp(max=6)) / 6) ** 2
    x = x * mask
    x = x.sum() / mask.sum()
    return x


def safe_div(numerator, denominator):
    if denominator.abs() < epsilon:
        if denominator < 0:
            denominator = denominator - epsilon
        else:
            denominator = denominator + epsilon
    return numerator / denominator


def get_smooth_f_score(input, target, beta=1.):
    true_positive = torch.min(input, target)
    false_positive = torch.relu(input - target)
    false_negative = torch.relu(target - input)

    TP = true_positive.sum()
    FP = false_positive.sum()
    FN = false_negative.sum()

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)

    beta2 = beta ** 2
    f_score = (1 + beta2) * safe_div(precision * recall, beta2 * precision + recall)

    return f_score, precision, recall


def get_notes_loss(input, target, beta=1):
    f_score = get_smooth_f_score(input, target, beta)[0]
    return 1. - f_score


def get_velocity_loss(input, target, mask):
    x = (target - input) ** 2
    x = x * mask
    return x.sum() / mask.sum()


def get_accidentals_loss(input, target, mask):
    x = F.binary_cross_entropy(input, target, reduction='none')
    x = x * mask.unsqueeze(-1)
    x = x.sum() / (mask.sum() * 3)
    return x


def get_song_info_loss(input, target):
    input_instruments, input_bpm, input_mode = input
    target_instruments, target_bpm, target_mode = target

    instruments_loss = F.binary_cross_entropy_with_logits(input_instruments, target_instruments)
    mode_loss = F.cross_entropy(input_mode, target_mode.argmax(1))
    bpm_loss = ((input_bpm - target_bpm) / bpm_range) ** 2
    return instruments_loss, mode_loss, bpm_loss


def get_channels_losses(input, target, pitched=True):
    target_velocity = get_velocity(target)
    mask = (target_velocity > 0).float()

    velocity = get_velocity(input)
    notes_loss = get_notes_loss(velocity, target_velocity)
    velocity_loss = get_velocity_loss(velocity, target_velocity, mask)
    duration_loss = get_duration_loss(get_duration(input), get_duration(target), mask)
    if pitched:
        accidentals_loss = get_accidentals_loss(
            get_accidentals(input), get_accidentals(target), mask)
        return notes_loss, velocity_loss, duration_loss, accidentals_loss
    return notes_loss, velocity_loss, duration_loss
