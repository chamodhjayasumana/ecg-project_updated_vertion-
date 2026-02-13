import numpy as np
from scipy.signal import welch
import pywt

def extract_window_features(window: np.ndarray, fs: int = 250):
    """
    window: shape [T]
    returns feature vector
    """

    # Time-domain
    mean = np.mean(window)
    std = np.std(window)
    energy = np.sum(window ** 2)

    # Frequency-domain (Welch PSD)
    freqs, psd = welch(window, fs=fs)
    lf = np.trapz(psd[(freqs >= 0.5) & (freqs <= 4)])
    hf = np.trapz(psd[(freqs > 4) & (freqs <= 40)])

    # Wavelet features (db4)
    coeffs = pywt.wavedec(window, "db4", level=3)
    wav_feats = []
    for c in coeffs:
        wav_feats.append(np.mean(c))
        wav_feats.append(np.std(c))

    return np.array([mean, std, energy, lf, hf] + wav_feats, dtype=np.float32)
