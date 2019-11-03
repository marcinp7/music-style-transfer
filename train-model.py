import math
import os

from flatten_dict import flatten as flatten_dict
import torch

from style.data import (
    iter_inputs,
    instrument_size,
    n_instruments,
    included_instruments,
    prepare_input,
    get_used_instruments,
)
from style.model import (
    device,
    get_total_loss,
    PitchedChannelsEncoder,
    UnpitchedChannelsEncoder,
    PitchedRhythmEncoder,
    UnpitchedRhythmEncoder,
    StyleEncoder,
    MelodyEncoder,
    SongInfoModel,
    PitchedStyleApplier,
    UnpitchedStyleApplier,
    StyleTransferModel,
)
from style.utils.data import save_to_csv, assert_dir
from style.utils.misc import iter_all_files, ProgressBar, dict_map
from style.utils.parallel import iter_parallel

data_path = 'data/Lakh MIDI Dataset/clean_midi/'

n_iterations = 5000
iter_size = 2
iter_size_increase_interval = 1000

training_info_path = 'training.csv'
save_path = 'snapshots/'
save_interval = 100

print(f'Using {device}')

print('Listing data files')

files = list(iter_all_files(data_path, '**/*.mid'))
train_files = files

print('Creating model')

torch.manual_seed(108)

beat_size = 64
bar_size = 128
n_rhythm_features = 8

style_size = 256
melody_size = 8
rhythm_size = 32

pitched_channels_encoder = PitchedChannelsEncoder(beat_size, bar_size, instrument_size).to(device)
unpitched_channels_encoder = UnpitchedChannelsEncoder(beat_size, bar_size).to(device)

pitched_rhythm_encoder = PitchedRhythmEncoder(
    rhythm_size, beat_size, bar_size, instrument_size).to(device)
unpitched_rhythm_encoder = UnpitchedRhythmEncoder(rhythm_size, beat_size, bar_size).to(device)

style_encoder = StyleEncoder(style_size, bar_size, instrument_size).to(device)
melody_encoder = MelodyEncoder(melody_size, beat_size, bar_size, instrument_size).to(device)

song_info_model = SongInfoModel(
    n_rhythm_features, style_size, rhythm_size, n_instruments).to(device)

pitched_style_applier = PitchedStyleApplier(
    style_size, melody_size, rhythm_size, instrument_size).to(device)
unpitched_style_applier = UnpitchedStyleApplier(style_size, rhythm_size).to(device)

model = StyleTransferModel(
    pitched_channels_encoder, unpitched_channels_encoder,
    style_encoder, melody_encoder,
    pitched_rhythm_encoder, unpitched_rhythm_encoder,
    song_info_model,
    pitched_style_applier, unpitched_style_applier,
)

print('Training')

optimizer = torch.optim.Adam(model.parameters(), lr=.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=.9)

inputs = iter_inputs(train_files, included_instruments, shuffle=True, looped=True)
inputs = iter_parallel(inputs)

optimizer.zero_grad()
pbar = ProgressBar(n_iterations)
for iteration in range(n_iterations):
    input = next(inputs)
    filename, (info, pitched_channels, instruments_features, instruments, unpitched_channels) =\
        input
    max_n_bars = 800 // pitched_channels.shape[0]
    mode, bpm, pitched_channels, instruments_features, unpitched_channels = prepare_input(
        input, max_n_bars)

    if pitched_channels.sum() == 0:
        continue
    if unpitched_channels is not None:
        if unpitched_channels.sum() == 0:
            unpitched_channels = None

    used_instruments = get_used_instruments(instruments_features, unpitched_channels)

    (instruments_pred, mode_pred, bpm_pred), pitched_pred, unpitched_pred =\
        model(mode, bpm, pitched_channels, instruments_features, unpitched_channels)
    losses = get_total_loss(
        instruments_pred, used_instruments,
        bpm_pred, info['bpm'],
        mode_pred, mode,
        pitched_pred, pitched_channels,
        unpitched_pred, unpitched_channels,
        normalize=True,
    )
    loss = losses['total']

    assert not math.isnan(loss)
    loss.backward()
    del loss

    losses = dict_map(lambda x: float(x) if x is not None else None, losses, recursive=True)
    pbar.add(
        1,
        total_loss=losses['total'],
        pitched_loss=losses['channels_loss']['pitched']['total'],
        pitched_notes_loss=losses['channels_loss']['pitched']['notes_loss'],
        song_info_loss=losses['song_info_loss']['total'],
        instruments_loss=losses['song_info_loss']['instruments_loss'],
        channelss_loss=losses['channels_loss']['total'],
        mode_loss=losses['song_info_loss']['mode_loss'],
        bpm_loss=losses['song_info_loss']['bpm_loss'],
    )
    if losses['channels_loss']['unpitched'] is not None:
        pbar.update_values(
            1,
            unpitched_loss=losses['channels_loss']['unpitched']['total'],
            unpitched_notes_loss=losses['channels_loss']['unpitched']['notes_loss'],
        )

    losses = flatten_dict(losses, reducer='underscore')
    save_to_csv(training_info_path, iteration=iteration, **losses)

    if (iteration + 1) % iter_size == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    if iteration % save_interval == 0:
        path = os.path.join(save_path, f'{iteration}.pkl')
        assert_dir(path)
        with open(path, 'wb') as f:
            torch.save(model, f)
