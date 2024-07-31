import torch
import torchaudio

import matplotlib.pyplot as plt
from torchaudio.io import StreamReader


streamer = StreamReader("test.wav")

#-noise = StreamReader(src="anoisesrc=color=white:sample_rate=16000:amplitude=0.5

# streamer.add_basic_audio_stream(
#     frames_per_chunk=8000,
#     sample_rate=8000,
# )

streamer.add_basic_audio_stream(
    frames_per_chunk=16000,
    sample_rate=16000,
)

n_ite = 3
waveforms = []
for i, (waveform,) in enumerate(streamer.stream()):
    waveforms.append(waveform)
    if i + 1 == n_ite:
        break

print(waveforms[0].shape)

k = 3
fig = plt.figure()
gs = fig.add_gridspec(3, k * n_ite)
for i, waveform in enumerate(waveforms):
    ax = fig.add_subplot(gs[0, k * i : k * (i + 1)])
    ax.specgram(waveform[:, 0], Fs=8000)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"Iteration {i}")
    if i == 0:
        ax.set_ylabel("Stream 0")

plt.tight_layout()
plt.show()
