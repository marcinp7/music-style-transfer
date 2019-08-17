# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from py_utils.pytorch import total_size, remove_dims, tensor_view, Distributed, squash_dims, LSTM

# from style.midi import get_input


def flatten_channel(x):
    x = remove_dims(x, 2)
    x = x.unsqueeze(2)
    return x


class ChannelEncoder(nn.Module):
    def __init__(self, n_channels=100, beats_lstm_size=100, bars_lstm_size=100):
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
            hidden_size=beats_lstm_size,
            num_layers=1,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=1)
        self.bars_lstm = LSTM(
            input_size=4*beats_lstm_size,
            hidden_size=bars_lstm_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        # input format: (batch, bar, beat, beat_fraction, note_features, note)
        x = squash_dims(x, 3, 5)  # combine beat fractions and note features
        x = self.beat_conv(x)
        x = F.relu(x)
        x = squash_dims(x, -2)
        x = self.beats_lstm(x)[0]
        x = squash_dims(x.contiguous(), 2)
        x = self.bars_lstm(x)[0]
        return x


class StyleEncoder(nn.Module):
    def __init__(self, input_size=200, hidden_size=100):
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = self.lstm(x)[0]
        x = x[:, -1]
        return x


class MelodyEncoder(nn.Module):
    def __init__(self, channel_encoder=None):
        super().__init__()
        self.channel_encoder = channel_encoder
        self.beat_conv = nn.Conv1d(
            in_channels=50,
            out_channels=7,
            kernel_size=14,
            stride=7,
            padding=4,
        )
        self.beat_conv = Distributed(self.beat_conv, depth=2)
        self.linear = nn.Linear(
            in_features=200,
            out_features=7,
        )

    def forward(self, channel, encoded_channel=None):
        channel = squash_dims(channel, 3, 5)  # combine beat fractions and note features
        x1 = self.beat_conv(channel)

        if encoded_channel is None:
            encoded_channel = self.channel_encoder(channel)
        x2 = self.linear(encoded_channel)
        x2 = x2.unsqueeze(-1).unsqueeze(2)

        x = x1 + x2
        x = squash_dims(x, -2)
        x = torch.sigmoid(x)
        # x = F.hardtanh(x, 0., 1.)

        x = x.unsqueeze(3)
        x = channel * x

        return x


class StyleApplier(nn.Module):
    def __init__(self, n_channels, beats_lstm_size, bars_lstm_size):
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
            hidden_size=beats_lstm_size,
            num_layers=1,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=1)
        self.bars_lstm = LSTM(
            input_size=4*beats_lstm_size,
            hidden_size=bars_lstm_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def duration_activation(self, x):
        return torch.relu(x)

    def velocity_activation(self, x):
        x = torch.relu(x)
        return torch.sigmoid(x)

    def apply_style(self, bar, style):
        n_beats = bar.shape[0]
        style = style.unsqueeze(0).expand(n_beats, -1, -1)
        concat_bar = torch.cat([bar, style], dim=2)
        output, _ = self.lstm(concat_bar)
        output = self.fc(output)
        return output

    def forward(self, melody, style):
        x = self.beat_conv(melody)
        x = F.relu(x)
        x = squash_dims(x, -2)
        x = self.beats_lstm(x)[0]
        x = squash_dims(x.contiguous(), 2)
        x = self.bars_lstm(x)[0]


        melody = flatten_channel(melody)
        bars = [self.apply_style(bar, style) for bar in melody]
        output = torch.stack(bars)
        output = tensor_view(output, keep_dims=2, *self.output_shape)

        durations = output[:, :, :, :, 0]
        velocities = output[:, :, :, :, 1]

        durations = self.duration_activation(durations)
        velocities = self.velocity_activation(velocities)

        output = torch.stack([durations, velocities], dim=-1)

        return output


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
