# from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import librosa
import torch
import torchaudio
# import torchaudio.functional as F
import torchaudio.transforms as T

import sounddevice as sd

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)

SAMPLE_MORSE = "/home/tristan/morse-code/generate/test.wav"


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower",
              aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


def play_audio(waveform, sample_rate=16000, format="s16"):
    # num_frames, num_channels = waveform.shape
    # s = torchaudio.io.StreamWriter(dst="-", format="alsa")
    # s.add_audio_stream(sample_rate, num_channels, format=)

    # with s.open():
    #     for i in range(0, num_frames, 256):
    #         s.write_audio_chunk(0, waveform[i : i + 256])
    sd.play(waveform, samplerate=sample_rate)


WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_MORSE)

spectrogram = T.Spectrogram(n_fft=512)

spec = spectrogram(WAVEFORM)

cmn = T.SlidingWindowCmn(cmn_window=1000)
cmn_waveform = cmn(WAVEFORM)

fig, axs = plt.subplots(2, 1)
plot_waveform(WAVEFORM, SAMPLE_RATE, title="Original waveform", ax=axs[0])
plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
fig.tight_layout()
plt.show()
