from src.dataloader.wavloader import MorseDataset
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

dataset = MorseDataset("/home/tristan/morse-code/model/labels.csv", "")

mfcc, label = dataset[0]
mfcc2, _ = dataset[4]

dataloader = DataLoader(dataset)

print(dataloader.collate_fn)

for X,y in dataloader:
    print(f"Shape of X [N, C, H, W]: {X.size()}")
    print(y)
    print(f"Shape of y: {y.size()} {y.dtype}")
    break

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

plot_spectrogram(mfcc[0], title="MFCC")
plot_spectrogram(mfcc2[0], title="MFCC2")
plt.show()
