import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

def bandpass_notch(x: np.ndarray, fs: float, bp=(0.5, 40.0), notch=50.0, q=30):
    x = x.astype(np.float32)
    b, a = butter(4, [bp[0]/(fs/2), bp[1]/(fs/2)], btype="band")
    y = filtfilt(b, a, x)
    b0, a0 = iirnotch(notch/(fs/2), q)
    y = filtfilt(b0, a0, y)
    return y.astype(np.float32)

def resample_to(x: np.ndarray, fs_in: float, fs_out: int):
    if int(round(fs_in)) == int(fs_out):
        return x.astype(np.float32), int(round(fs_in))
    y = resample_poly(x, fs_out, int(round(fs_in)))
    return y.astype(np.float32), fs_out

def z_norm(seg: np.ndarray):
    return ((seg - seg.mean()) / (seg.std() + 1e-8)).astype(np.float32)
