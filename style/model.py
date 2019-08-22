import torch
from torch import nn

from style.utils.pytorch import Distributed, squash_dims, LSTM


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
        # (batch, bar, beat, beat_fraction, note_features, note)
        if len(x.shape) == 6:
            x = squash_dims(x, 3, 5)  # (batch, bar, beat, features, note)
        x = self.beat_conv(x)  # (batch, bar, beat, features, octave)
        x = torch.relu(x)
        x = squash_dims(x, -2)  # (batch, bar, beat, features)
        beats = self.beats_lstm(x)[0]  # (batch, bar, beat, features)
        x = beats[:, :, -1]  # (batch, bar, features)
        bars = self.bars_lstm(x)[0]  # (batch, bar, features)
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
        # input format: (batch, bar, features)
        x = self.lstm(bars)[0]  # (batch, bar, features)
        x = x[:, -1]  # (batch, features)
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
            channel = squash_dims(channel, 3, 5)  # (batch, bar, beat, features, note)
        x1 = self.beat_conv(channel)  # (batch, bar, beat, scale_degree, octave)

        x2 = self.beats_linear(beats)  # (batch, bar, beat, scale_degree)
        x2 = x2.unsqueeze(-1)  # (batch, bar, beat, scale_degree, octave)

        x3 = self.bars_linear(bars)  # (batch, bar, scale_degree)
        x3 = x3.unsqueeze(-1).unsqueeze(2) # (batch, bar, beat, scale_degree, octave)

        x = x1 + x2 + x3 # (batch, bar, beat, scale_degree, octave)
        # octave must come before scale degree
        x = x.transpose(3, 4).contiguous() # (batch, bar, beat, octave, scale_degree)
        x = squash_dims(x, -2) # (batch, bar, beat, note)
        x = torch.sigmoid(x)
        # x = F.hardtanh(x, 0., 1.)

        x = x.unsqueeze(3) # (batch, bar, beat, features, note)
        x = channel * x  # (batch, bar, beat, features, note)
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
        x = torch.tanh(x)
        x = torch.relu(x)
        return x

    @classmethod
    def accidentals_activation(cls, x):
        x = torch.sigmoid(x)
        return x

    def forward(self, melody, style):
        beats, bars = self.channel_encoder(melody)

        x1 = self.beats_linear(beats) # (batch, bar, beat, features)

        x2 = self.bars_linear(bars) # (batch, bar, features)
        x2 = x2.unsqueeze(2) # (batch, bar, beat, features)

        x3 = self.style_linear(style) # (batch, features)
        x3 = x3.unsqueeze(1).unsqueeze(1) # (batch, bar, beat, features)

        x4 = self.beats_conv(melody) # (batch, bar, beat, features, octave)
        # (batch, bar, beat, note_fraction, note_features, scale_degree, octave)
        x4 = x4.view(*x4.shape[:3], 10, 5, 7, 8)
        # octave must come before scale degree
        x4 = x4.transpose(-1, -2)
        # (batch, bar, beat, note_fraction, note_features, octave, scale_degree)
        x4 = squash_dims(x4.contiguous(), -2)
        # (batch, bar, beat, note_fraction, note_features, note)

        x = x1 + x2 + x3
        x = x.view(*x.shape[:3], 10, 5, -1) # (batch, bar, beat, beat_fraction, note_features, note)
        x += x4

        duration = self.duration_activation(x[:, :, :, :, :1])
        velocity = self.velocity_activation(x[:, :, :, :, 1:2])
        accidentals = self.accidentals_activation(x[:, :, :, :, 2:])
        x = torch.cat([duration, velocity, accidentals], 4)
        # (batch, bar, beat, beat_fraction, note_features, note)
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


def hard_output(x):
    accidentals = x[:, :, :, :, 2:]
    accidentals = accidentals > .5
    x = torch.cat([x[:, :, :, :, :2], accidentals.float()], 4)
    return x


def get_duration(x):
    return x[:, :, :, :, 0]


def get_velocity(x):
    return x[:, :, :, :, 1]


def get_accidentals(x):
    return x[:, :, :, :, 2:]


def get_duration_loss(input, target):
    x = torch.log((1. + input) / (1. + target)) ** 2
    return x.mean()


def get_velocity_loss(input, target):
    error = (target - input) ** 2

    positives = (target > 0).float()
    negatives = (target == 0).float()

    false_positives = error * negatives
    false_negatives = error * positives

    false_positives_error = false_positives.sum() / (1. + negatives.sum())
    false_negatives_error = false_negatives.sum() / (1. + positives.sum())

    total_loss = .5 * false_positives_error + .5 * false_negatives_error

    return total_loss


epsilon = 1e-6


def get_accidentals_loss(input, target):
    x = target * torch.log(input + epsilon) + (1. - target) * torch.log(1. - input + epsilon)
    return -x.mean()


def get_loss(input, target):
    duration_loss = get_duration_loss(get_duration(input), get_duration(target))
    velocity_loss = get_velocity_loss(get_velocity(input), get_velocity(target))
    accidentals_loss = get_accidentals_loss(get_accidentals(input), get_accidentals(target))
    total_loss = .1 * duration_loss + .1 * accidentals_loss + .8 * velocity_loss
    return total_loss
