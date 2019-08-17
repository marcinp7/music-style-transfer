# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from py_utils.pytorch import remove_dims, tensor_view, Distributed, squash_dims, LSTM

# from style.midi import get_input


def flatten_channel(x):
    x = remove_dims(x, 2)
    x = x.unsqueeze(2)
    return x


class ChannelEncoder(nn.Module):
    def __init__(self, n_channels=50, beat_size=16, bar_size=32):
        super().__init__()
        self.beat_conv = nn.Conv1d(
            in_channels=50,
            out_channels=n_channels,
            kernel_size=14,
            stride=7,
            padding=4,
        )
        self.beat_conv = Distributed(self.beat_conv, depth=2)
        self.beats_lstm = LSTM(
            input_size=8*n_channels,
            hidden_size=beat_size,
            num_layers=1,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=1)
        self.bars_lstm = LSTM(
            input_size=4*beat_size,
            hidden_size=bar_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        # input format: (batch, bar, beat, beat_fraction, note_features, note)
        if len(x.shape) == 6:
            x = squash_dims(x, 3, 5)  # combine beat fractions and note features
        x = self.beat_conv(x)
        x = F.relu(x)
        x = squash_dims(x, -2)
        beats = self.beats_lstm(x)[0]
        x = squash_dims(beats.contiguous(), 2)
        bars = self.bars_lstm(x)[0]
        return beats, bars


class StyleEncoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=100):
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, bars):
        x = self.lstm(bars)[0]
        x = x[:, -1]
        return x


class MelodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.beat_conv = nn.Conv1d(
            in_channels=50,
            out_channels=7,
            kernel_size=14,
            stride=7,
            padding=4,
        )
        self.beat_conv = Distributed(self.beat_conv, depth=2)
        self.beats_linear = nn.Linear(
            in_features=32,
            out_features=7,
        )
        self.bars_linear = nn.Linear(
            in_features=64,
            out_features=7,
        )

    def forward(self, channel, beats, bars):
        if len(channel.shape) == 6:
            channel = squash_dims(channel, 3, 5)  # combine beat fractions and note features
        x1 = self.beat_conv(channel)

        x2 = self.beats_linear(beats)
        x2 = x2.unsqueeze(-1)

        x3 = self.bars_linear(bars)
        x3 = x3.unsqueeze(-1).unsqueeze(2)

        x = x1 + x2 + x3
        x = x.transpose(3, 4).contiguous() # octave must come before scale degree
        x = squash_dims(x, -2)
        x = torch.sigmoid(x)
        # x = F.hardtanh(x, 0., 1.)

        x = x.unsqueeze(3)
        x = channel * x

        return x


class StyleApplier(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_encoder = ChannelEncoder(n_channels=10, beat_size=4, bar_size=8)
        out_features = 10 * 5 * 8 * 7
        self.beats_linear = nn.Linear(
            in_features=4,
            out_features=out_features,
        )
        self.bars_linear = nn.Linear(
            in_features=16,
            out_features=out_features,
        )
        self.style_linear = nn.Linear(
            in_features=100,
            out_features=out_features,
        )
        self.beats_conv = nn.Conv1d(
            in_channels=50,
            out_channels=7*50,
            kernel_size=14,
            stride=7,
            padding=4,
        )
        self.beats_conv = Distributed(self.beats_conv, depth=2)

    @classmethod
    def duration_activation(cls, x):
        x = torch.relu(x)
        return x

    @classmethod
    def velocity_activation(cls, x):
        x = torch.relu(x)
        return x

    @classmethod
    def accidental_activation(cls, x):
        x = torch.sigmoid(x)
        return x

    def forward(self, melody, style):
        beats, bars = self.channel_encoder(melody)

        x1 = self.beats_linear(beats)
        # print(x1.shape)

        x2 = self.bars_linear(bars)
        x2 = x2.unsqueeze(2)
        # print(x2.shape)

        x3 = self.style_linear(style)
        x3 = x3.unsqueeze(1).unsqueeze(1)
        # print(x3.shape)

        x4 = self.beats_conv(melody)
        x4 = x4.view(*x4.shape[:3], 10, 5, 7, 8)
        x4 = x4.transpose(-1, -2) # octave must come before scale degree
        # print(x4.shape)

        x = x1 + x2 + x3
        x = x.view(*x.shape[:3], 10, 5, 8, 7)
        x += x4
        x = squash_dims(x, -2)
        # print(x.shape)

        x[:, :, :, :, 0, :] = self.duration_activation(x[:, :, :, :, 0, :])
        x[:, :, :, :, 1, :] = self.velocity_activation(x[:, :, :, :, 1, :])
        x[:, :, :, :, 2:, :] = self.accidental_activation(x[:, :, :, :, 2:, :])

        return x


class StyleTransferModel(nn.Module):
    def __init__(self, channel_encoder, melody_encoder, style_encoder, style_applier):
        super().__init__()
        self.channel_encoder = channel_encoder
        self.melody_encoder = melody_encoder
        self.style_encoder = style_encoder
        self.style_applier = style_applier

    def prepare_input(self, vchannel, device='cpu'):
        return self.channel_encoder.prepare_input(vchannel, device)

    def forward(self, x):
        encoded = self.channel_encoder(x)
        melody = self.melody_encoder(x, encoded)
        style = self.style_encoder(encoded)
        applied = self.style_applier(melody, style)
        return applied


def get_duration(y):
    return y[:, :, :, :, 0]


def get_velocity(y):
    return y[:, :, :, :, 1]


def duration_loss_func(y, pred):
    return torch.log((1 + y) / (1 + pred)) ** 2


def velocity_loss_func(y, pred):
    return (y - pred) ** 2


def loss_func(y, pred):
    duration_loss = duration_loss_func(get_duration(y), get_duration(pred))
    velocity_loss = velocity_loss_func(get_velocity(y), get_velocity(pred))
    total_loss = duration_loss + velocity_loss
    return total_loss.mean()
