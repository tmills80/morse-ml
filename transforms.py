import torch
import torchaudio.transforms as T


class FeatureExtraction(torch.nn.Module):
    def __init__(
            self,
            input_freq=8000,
            resample_freq=8000,
            n_fft=1024,
            n_mel=256,
            fade=None,
            noise_add=None,
            ):
        super().__init__()
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)

        self.spec = T.Spectrogram(n_fft=n_fft, power=2)

        

