import torch
from torch import nn
# import torch.nn.functional as F

from py_utils.pytorch import Distributed, squash_dims, LSTM


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
            input_size=beat_size,
            hidden_size=bar_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    # todo: improve encoding beats
    def forward(self, x):
        # input format: (batch, bar, beat, beat_fraction, note_features, note)
        if len(x.shape) == 6:
            x = squash_dims(x, 3, 5)  # combine beat fractions and note features
        x = self.beat_conv(x)
        x = torch.relu(x)
        x = squash_dims(x, -2)
        beats = self.beats_lstm(x)[0]
        x = beats[:, :, -1]
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
    def accidentals_activation(cls, x):
        x = torch.sigmoid(x)
        return x

    def forward(self, melody, style):
        beats, bars = self.channel_encoder(melody)

        x1 = self.beats_linear(beats)

        x2 = self.bars_linear(bars)
        x2 = x2.unsqueeze(2)

        x3 = self.style_linear(style)
        x3 = x3.unsqueeze(1).unsqueeze(1)

        x4 = self.beats_conv(melody)
        x4 = x4.view(*x4.shape[:3], 10, 5, 7, 8)
        x4 = x4.transpose(-1, -2) # octave must come before scale degree

        x = x1 + x2 + x3
        x = x.view(*x.shape[:3], 10, 5, 8, 7)
        x += x4
        x = squash_dims(x, -2)

        duration = self.duration_activation(x[:, :, :, :, :1])
        velocity = self.velocity_activation(x[:, :, :, :, 1:2])
        accidentals = self.accidentals_activation(x[:, :, :, :, 2:])
        x = torch.cat([duration, velocity, accidentals], 4)
        return x


class StyleTransferModel(nn.Module):
    def __init__(self, channel_encoder, melody_encoder, style_encoder, style_applier):
        super().__init__()
        self.channel_encoder = channel_encoder
        self.melody_encoder = melody_encoder
        self.style_encoder = style_encoder
        self.style_applier = style_applier

    def forward(self, x):
        beats, bars = self.channel_encoder(x)
        melody = self.melody_encoder(x, beats, bars)
        style = self.style_encoder(bars)
        x = self.style_applier(melody, style)
        return x


def get_duration(x):
    return x[:, :, :, :, 0]


def get_velocity(x):
    return x[:, :, :, :, 1]


def get_accidentals(x):
    return x[:, :, :, :, 2:]


def get_duration_loss(input, target):
    x = torch.log((1 + target) / (1 + input)) ** 2
    return x.mean()


def get_velocity_loss(input, target):
    x = (target - input) ** 2
    return x.mean()


epsilon = 1e-6


def get_accidentals_loss(input, target):
    x = target * torch.log(input + epsilon) + (1. - target) * torch.log(1. - input + epsilon)
    return -x.mean()


def get_loss(input, target):
    duration_loss = get_duration_loss(get_duration(target), get_duration(input))
    velocity_loss = get_velocity_loss(get_velocity(target), get_velocity(input))
    accidentals_loss = get_accidentals_loss(get_accidentals(target), get_accidentals(input))
    total_loss = duration_loss + velocity_loss + accidentals_loss
    return total_loss
