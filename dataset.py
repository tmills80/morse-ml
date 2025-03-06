import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import src.utils as utils

SAMPLE_RATE=8000


class MorseDataset(Dataset):
    """
    Dataset which reads .wav file and returns the MFCC of the wav
    """

    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256

    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )

    label_transform = utils.StringToTensor()

    def __init__(self, labels_file, data_dir):
        self.labels = pd.read_csv(labels_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item_path = self.labels.iloc[idx, 0] # labels has full path

        waveform = torchaudio.load(item_path)

        sample_rate = waveform[1]

        if sample_rate != SAMPLE_RATE:
            wf = waveform[0]
            waveform = (
                T.Resample(sample_rate, SAMPLE_RATE, dtype=wf.dtype)(wf),
                SAMPLE_RATE
            )

        label = self.label_transform(self.labels.iloc[idx, 1][1:-1])


        mfcc = self.mfcc_transform(waveform[0])

        return mfcc, label






# dataset = MorseDataset("/home/tristan/morse-code/model/labels.csv", "")

# mfcc, _ = dataset[1]
# mfcc2, _ = dataset[4]



# def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
#     if ax is None:
#         _, ax = plt.subplots(1, 1)
#     if title is not None:
#         ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

# plot_spectrogram(mfcc[0], title="MFCC")
# plot_spectrogram(mfcc2[0], title="MFCC2")
# plt.show()
