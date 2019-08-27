import torch
from torch import nn

from style.utils.pytorch import Distributed, squash_dims, LSTM


class PitchedChannelsEncoder(nn.Module):
    def __init__(self, instrument_size=4, n_conv_channels=50, beat_size=16, bar_size=32):
        super().__init__()
        self.instruments_linear = nn.Linear(
            in_features=instrument_size,
            out_features=8*n_conv_channels,
        )
        self.beat_conv = nn.Conv1d(
            in_channels=10*5,
            out_channels=n_conv_channels,
            kernel_size=14,
            stride=7,
            padding=4,
        )
        self.beat_conv = Distributed(self.beat_conv, depth=3)
        self.beats_lstm = LSTM(
            input_size=8*n_conv_channels,
            hidden_size=beat_size,
            num_layers=1,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=2)
        self.bars_lstm = LSTM(
            input_size=beat_size,
            hidden_size=bar_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.bars_lstm = Distributed(self.bars_lstm, depth=1)

    def forward(self, channels, instruments):
        # channels: (batch, channel, bar, beat, beat_fraction, note, note_features)
        # instruments: (batch, channel, features)

        x = channels.transpose(-1, -2)
        # (batch, channel, bar, beat, beat_fraction, note_features, note)
        x = x.contiguous()
        x = squash_dims(x, 4, 6)  # (batch, channel, bar, beat, features, note)
        x = self.beat_conv(x)  # (batch, channel, bar, beat, features, octave)
        x = torch.relu(x)
        x1 = squash_dims(x, -2)  # (batch, channel, bar, beat, features)

        x = self.instruments_linear(instruments)  # (batch, channel, features)
        x2 = x.unsqueeze(2).unsqueeze(2)  # (batch, channel, bar, beat, features)

        x = x1 + x2  # (batch, channel, bar, beat, features)
        x = torch.relu(x)

        beats = self.beats_lstm(x)[0]  # (batch, channel, bar, beat, features)
        x = beats[:, :, :, -1]  # (batch, channel, bar, features)
        bars = self.bars_lstm(x)[0]  # (batch, channel, bar, features)
        return beats, bars


class UnpitchedChannelsEncoder(nn.Module):
    def __init__(self, beat_size=16, bar_size=32):
        super().__init__()
        self.beat_linear = nn.Linear(
            in_features=10*47*2,
            out_features=100,
        )
        self.beats_lstm = LSTM(
            input_size=100,
            hidden_size=beat_size,
            num_layers=1,
            batch_first=True,
        )
        self.beats_lstm = Distributed(self.beats_lstm, depth=2)
        self.bars_lstm = LSTM(
            input_size=beat_size,
            hidden_size=bar_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.bars_lstm = Distributed(self.bars_lstm, depth=1)

    def forward(self, channels):
        # channels: (batch, channel, bar, beat, beat_fraction, note, note_features)
        x = channels.transpose(-1, -2)
        # (batch, channel, bar, beat, beat_fraction, note_features, note)
        x = x.contiguous()
        x = squash_dims(x, 4, 7)  # (batch, channel, bar, beat, features)
        x = self.beat_linear(x)  # (batch, channel, bar, beat, features)
        x = torch.relu(x)
        beats = self.beats_lstm(x)[0]  # (batch, channel, bar, beat, features)
        x = beats[:, :, :, -1]  # (batch, channel, bar, features)
        bars = self.bars_lstm(x)[0]  # (batch, channel, bar, features)
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
        self.lstm = Distributed(self.lstm, depth=1)

    def forward(self, bars):
        x = self.lstm(bars)[0]  # (batch, channel, bar, features)
        x = x[:, :, -1]  # (batch, channel, features)
        x = combine(x, dim=1)  # (batch, features)
        return x


class RhythmEncoder(nn.Module):
    def __init__(self, n_notes, n_note_features, rhythm_size=32):
        super().__init__()
        self.channels_linear = nn.Linear(
            in_features=n_notes*n_note_features,
            out_features=rhythm_size,
        )
        self.beats_linear = nn.Linear(
            in_features=16,
            out_features=rhythm_size,
        )
        self.bars_linear = nn.Linear(
            in_features=64,
            out_features=rhythm_size,
        )

    def forward(self, channels, beats, bars):
        x = squash_dims(channels, -2)  # (batch, channel, bar, beat, beat_fraction, features)
        x1 = self.channels_linear(x)  # (batch, channel, bar, beat, beat_fraction, features)

        x = self.beats_linear(beats)  # (batch, channel, bar, beat, features)
        x2 = x.unsqueeze(4)  # (batch, channel, bar, beat, beat_fraction, features)

        x = self.bars_linear(bars)  # (batch, channel, bar, features)
        x3 = x.unsqueeze(3).unsqueeze(4)  # (batch, channel, bar, beat, beat_fraction, features)

        x = x1 + x2 + x3  # (batch, channel, bar, beat, beat_fraction, features)
        x = torch.relu(x)
        x = combine(x, dim=1)  # (batch, bar, beat, beat_fraction, features)
        return x


class MelodyEncoder(nn.Module):
    def __init__(self, melody_size=32):
        super().__init__()
        self.beat_conv = nn.Conv1d(
            in_channels=5*10,
            out_channels=7,
            kernel_size=14,
            stride=7,
            padding=4,
        )
        self.beat_conv = Distributed(self.beat_conv, depth=3)
        self.beats_linear = nn.Linear(
            in_features=16,
            out_features=7,
        )
        self.bars_linear = nn.Linear(
            in_features=64,
            out_features=7,
        )
        self.melody_linear = nn.Linear(
            in_features=5,
            out_features=melody_size,
        )

    def forward(self, channels, beats, bars):
        x = channels.transpose(-1, -2)
        # (batch, channel, bar, beat, beat_fraction, note_features, note)
        x = x.contiguous()
        x = squash_dims(x, 4, 6)  # (batch, channel, bar, beat, features, note)
        x1 = self.beat_conv(x)  # (batch, channel, bar, beat, scale_degree, octave)

        x = self.beats_linear(beats)  # (batch, channel, bar, beat, scale_degree)
        x2 = x.unsqueeze(-1)  # (batch, channel, bar, beat, scale_degree, octave)

        x = self.bars_linear(bars)  # (batch, channel, bar, scale_degree)
        x3 = x.unsqueeze(-1).unsqueeze(3)  # (batch, channel, bar, beat, scale_degree, octave)

        x = x1 + x2 + x3  # (batch, channel, bar, beat, scale_degree, octave)
        # octave must come before scale degree
        x = x.transpose(4, 5).contiguous()  # (batch, channel, bar, beat, octave, scale_degree)
        x = squash_dims(x, -2)  # (batch, channel, bar, beat, note)
        x = torch.sigmoid(x)
        x = x.unsqueeze(-1).unsqueeze(4)
        # (batch, channel, bar, beat, beat_fraction, note, features)
        x = channels * x  # (batch, channel, bar, beat, beat_fraction, note, note_features)
        x = self.melody_linear(x)  # (batch, channel, bar, beat, beat_fraction, note, note_features)
        x = combine(x, dim=1)  # (batch, bar, beat, beat_fraction, note, note_features)
        return x


class PitchedStyleApplier(nn.Module):
    def __init__(self, melody_size=32, instrument_size=4):
        super().__init__()
        self.style_linear_degrees = nn.Linear(
            in_features=100,
            out_features=10*7*melody_size,
        )
        self.style_linear_octaves = nn.Linear(
            in_features=100,
            out_features=10*8*melody_size,
        )
        self.instruments_linear = nn.Linear(
            in_features=instrument_size,
            out_features=melody_size,
        )
        self.linear = nn.Linear(
            in_features=melody_size,
            out_features=5,
        )

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

    def forward(self, style, melody, rhythm, instruments):
        x = self.style_linear_degrees(style)  # (batch, features)
        x1 = x.view(x.size(0), 1, 1, 1, 10, 1, 7, -1)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x = self.style_linear_octaves(style)  # (batch, features)
        x2 = x.view(x.size(0), 1, 1, 1, 10, 8, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x3 = melody.view(*melody.shape[:4], 8, 7, -1)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x4 = rhythm.view(rhythm.shape[0], 1, *rhythm.shape[1:4], 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x = self.instruments_linear(instruments)  # (batch, channel, features)
        x5 = x.view(*x.shape[:2], 1, 1, 1, 1, 1, -1)
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)

        x = x1 + x2 + x3 + x4 + x5
        # (batch, channel, bar, beat, beat_fraction, octave, scale_degree, features)
        x = squash_dims(x, 5, 7)  # (batch, channel, bar, beat, beat_fraction, note, features)
        # x = torch.relu(x)
        x = self.linear(x)  # (batch, channel, bar, beat, beat_fraction, note, note_features)

        duration = self.duration_activation(x[:, :, :, :, :, :1])
        velocity = self.velocity_activation(x[:, :, :, :, :, 1:2])
        accidentals = self.accidentals_activation(x[:, :, :, :, :, 2:])
        x = torch.cat([duration, velocity, accidentals], 5)
        # (batch, channel, bar, beat, beat_fraction, note, note_features)
        return x


class UnpitchedStyleApplier(nn.Module):
    def __init__(self, rhythm_size=32):
        super().__init__()
        self.style_linear = nn.Linear(
            in_features=100,
            out_features=10*47*rhythm_size,
        )
        self.linear = nn.Linear(
            in_features=rhythm_size,
            out_features=2,
        )

    @classmethod
    def duration_activation(cls, x):
        x = torch.relu(x)
        return x

    @classmethod
    def velocity_activation(cls, x):
        x = torch.tanh(x)
        x = torch.relu(x)
        return x

    def forward(self, style, rhythm):
        x = self.style_linear(style)  # (batch, features)
        x1 = x.view(x.size(0), 1, 1, 10, 47, -1)
        # (batch, bar, beat, beat_fraction, note, features)

        x2 = rhythm.view(*rhythm.shape[:4], 1, -1)
        # (batch, bar, beat, beat_fraction, note, features)

        x = x1 + x2
        # (batch, bar, beat, beat_fraction, note, features)
        # x = torch.relu(x)
        x = self.linear(x)  # (batch, bar, beat, beat_fraction, note, note_features)

        duration = self.duration_activation(x[:, :, :, :, :, :1])
        velocity = self.velocity_activation(x[:, :, :, :, :, 1:2])
        x = torch.cat([duration, velocity], 5)
        # (batch, bar, beat, beat_fraction, note, note_features)
        return x


class StyleTransferModel(nn.Module):
    def __init__(self, pitched_channels_encoder, unpitched_channels_encoder, style_encoder,
                 melody_encoder, pitched_rhythm_encoder, unpitched_rhythm_encoder,
                 pitched_style_applier, unpitched_style_applier):
        super().__init__()
        self.pitched_channels_encoder = pitched_channels_encoder
        self.unpitched_channels_encoder = unpitched_channels_encoder

        self.style_encoder = style_encoder
        self.melody_encoder = melody_encoder

        self.pitched_rhythm_encoder = pitched_rhythm_encoder
        self.unpitched_rhythm_encoder = unpitched_rhythm_encoder

        self.pitched_style_applier = pitched_style_applier
        self.unpitched_style_applier = unpitched_style_applier

    def forward(self, pitched_channels, instruments, unpitched_channels):
        pitched_beats, pitched_bars = self.pitched_channels_encoder(pitched_channels, instruments)
        unpitched_beats, unpitched_bars = self.unpitched_channels_encoder(unpitched_channels)
        bars = torch.cat([pitched_bars, unpitched_bars], 1)

        style = self.style_encoder(bars)
        melody = self.melody_encoder(pitched_channels, pitched_beats, pitched_bars)

        pitched_rhythm = self.pitched_rhythm_encoder(pitched_channels, pitched_beats, pitched_bars)
        unpitched_rhythm = self.unpitched_rhythm_encoder(unpitched_channels, unpitched_beats,
                                                         unpitched_bars)
        rhythm = combine(pitched_rhythm, unpitched_rhythm)

        x_pitched = self.pitched_style_applier(style, melody, rhythm, instruments)
        x_unpitched = self.unpitched_style_applier(style, rhythm)

        return x_pitched, x_unpitched


def combine(*tensors, dim=None):
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
    norm = torch.sqrt(x)

    tensor *= norm
    return tensor.sum(dim) / norm.sum()


def hard_output(x):
    duration = x[:, :, :, :, :, :1]
    velocity = x[:, :, :, :, :, 1:2]
    accidentals = x[:, :, :, :, :, 2:]

    velocity *= (velocity > .05).float()

    max_accidentals = accidentals.max(dim=-1)[0]
    new_accidentals = accidentals == max_accidentals.unsqueeze(-1)
    new_accidentals *= accidentals > .1
    x = torch.cat([duration, velocity, new_accidentals.float()], 5)
    return x


def get_duration(x):
    return x[:, :, :, :, :, 0]


def get_velocity(x):
    return x[:, :, :, :, :, 1]


def get_accidentals(x):
    return x[:, :, :, :, :, 2:]


def get_duration_loss(input, target, mask):
    # x = (torch.log((1. + input) / (1. + target)) * mask) ** 2
    x = (torch.sigmoid(input) - torch.sigmoid(target)) ** 2
    x = x.sum() / mask.sum()
    return x


def get_smooth_f1_score(target, error):
    positive = target
    negative = 1. - positive

    false_positive = error * negative
    false_negative = error * positive

    true_positive = (1. - error) * positive
    # true_negative = (1. - error) * negative

    TP = true_positive.sum()
    FP = false_positive.sum()
    FN = false_negative.sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score, precision, recall


def get_notes_loss(input, target):
    error = (target - input) ** 2
    f1_score = get_smooth_f1_score(target, error)[0]
    return 1. - f1_score


def get_velocity_loss(input, target, mask):
    x = (target - input) ** 2
    x *= mask
    return x.sum() / mask.sum()


def get_accidentals_loss(input, target, mask):
    # x = nn.functional.binary_cross_entropy(input, target, reduction='none')
    x = (input - target) ** 2
    x *= mask.unsqueeze(-1)
    x = x.sum() / (mask.sum() * 3)
    return x


def get_losses(input, target):
    target_velocity = get_velocity(target)
    mask = (target_velocity > 0.).float()

    velocity = get_velocity(input)
    notes_loss = get_notes_loss(velocity, target_velocity)
    velocity_loss = get_velocity_loss(velocity, target_velocity, mask)

    duration_loss = get_duration_loss(get_duration(input), get_duration(target), mask)
    accidentals_loss = get_accidentals_loss(get_accidentals(input), get_accidentals(target), mask)
    return notes_loss, velocity_loss, accidentals_loss, duration_loss
