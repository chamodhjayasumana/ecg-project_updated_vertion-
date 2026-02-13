import numpy as np
import pywt

def window_to_scalogram_cwt(
    w: np.ndarray,
    fs: int = 250,
    wavelet: str = "morl",
    n_scales: int = 64,
    f_min: float = 0.5,
    f_max: float = 40.0,
) -> np.ndarray:
    """
    Convert 1D window -> CWT scalogram (n_scales x T).
    Uses frequency-guided scales so your bands match ECG.
    """
    w = np.asarray(w, dtype=np.float32)
    w = w - w.mean()

    # Build target frequencies and convert to scales
    freqs = np.linspace(f_min, f_max, n_scales)
    fc = pywt.central_frequency(wavelet)
    scales = fc * fs / freqs  # scale = fc * fs / f

    coefs, _ = pywt.cwt(w, scales, wavelet, sampling_period=1.0/fs)
    S = np.abs(coefs).astype(np.float32)  # magnitude scalogram

    # Log + normalize (helps CNN)
    S = np.log1p(S)
    S = (S - S.mean()) / (S.std() + 1e-6)
    return S  # shape: [n_scales, T]
