import numpy as np
import torch

from py_utils.pytorch import total_size, remove_dims, tensor_view

from style.midi import get_input


def get_inputs(files, shuffle=False):
    if shuffle:
        files = files[:]
        np.random.shuffle(files)
    for file in files:
        yield get_input(file)


class ChannelEncoder(torch.nn.Module):
    def __init__(self, input_shape, local_bar_size=20, bar_size=60, num_layers=1):
        super().__init__()
        self.input_shape = input_shape
        self.local_bar_size = local_bar_size
        self.bar_size = bar_size
        self.num_layers = num_layers

        self.beats_lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.local_bar_size,
            num_layers=self.num_layers,
        )

        self.bars_lstm = torch.nn.LSTM(
            input_size=self.local_bar_size,
            hidden_size=self.local_bar_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )

        self.fc = torch.nn.Linear(local_bar_size * 2, self.bar_size)

    @property
    def input_size(self):
        return total_size(self.input_shape)

    def bar2tensor(self, bar, device):
        bar = [torch.tensor(beat, dtype=torch.float) for beat in bar]
        bar = torch.stack(bar)
        bar = bar.to(device)
        return bar

    def get_bar_vector(self, bar):
        output, _ = self.beats_lstm(bar)
        return output[-1]

    def forward(self, input):
        bar_vectors = [self.get_bar_vector(bar) for bar in input]
        bar_vectors = torch.stack(bar_vectors)
        bar_vectors, _ = self.bars_lstm(bar_vectors)
        bar_vectors = [self.fc(bar_vector) for bar_vector in bar_vectors]
        bar_vectors = torch.stack(bar_vectors)
        return bar_vectors

    def prepare_input(self, vchannel, device='cpu'):
        bars = [self.bar2tensor(bar, device) for bar in vchannel]
        bars = torch.stack(bars)
        return bars

    def encode_vchannel(self, vchannel, device):
        x = self.prepare_input(vchannel, device)
        return self.forward(x)


class StyleEncoder(torch.nn.Module):
    def __init__(self, bar_size, style_size=300, num_layers=1):
        super().__init__()
        self.bar_size = bar_size
        self.style_size = style_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            input_size=self.bar_size,
            hidden_size=self.style_size,
            num_layers=self.num_layers,
        )

    def forward(self, input):
        output, _ = self.lstm(input)
        return output[-1]


class MelodyEncoder(torch.nn.Module):
    def __init__(self, channel_encoder, output_shape, num_layers=2):
        super().__init__()
        self.channel_encoder = channel_encoder
        self.output_shape = output_shape
        self.num_layers = num_layers

        lstm_input_size = self.channel_encoder.bar_size + self.channel_encoder.input_size
        self.lstm = torch.nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.output_size,
            num_layers=self.num_layers,
        )

    @property
    def output_size(self):
        return total_size(self.output_shape)

    def get_bar_melody(self, bar):
        output, _ = self.lstm(bar)
        return output

    def forward(self, channel, encoded_channel=None):
        if encoded_channel is None:
            encoded_channel = self.channel_encoder(channel)
        n_beats = channel.shape[1]
        expanded_channel = encoded_channel.unsqueeze(2).expand(-1, n_beats, -1, -1)
        concat_channel = torch.cat([channel, expanded_channel], 3)

        bars = [self.get_bar_melody(bar) for bar in concat_channel]
        output = torch.stack(bars)
        return tensor_view(output, *self.output_shape, keep_dims=2)

    def encode_melody(self, vchannel, device):
        channel = self.channel_encoder.prepare_input(vchannel, device)
        return self.forward(channel)
