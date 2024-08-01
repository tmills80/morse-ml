from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import pandas as pd
import os


class MorseDataset(Dataset):
    """
    Dataset implementation to return audio files.
    Resamples to <target_sr> if necessary (default 8000)
    Resampling may be expensive, so avoid if possible
    """

    def __init__(self, labels_file, data_dir, transform=None,
                 target_tranform=None, target_sr=8000):
        self.labels = pd.read_csv(labels_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_tranform = target_tranform
        self.target_sr = target_sr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.target_sr:
            waveform = F.resample(waveform, sample_rate,
                                  self.target_sr)

        label = self.labels.iloc[idx, 1]
        if self.transform:
            waveform = self.transform(waveform)
        if self.target_transform:
            label = self.target_tranform(label)

        return waveform, label
