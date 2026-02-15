import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from wavelet_utils import window_to_scalogram_cwt


def main():
    # Paths
    data_dir = Path("../data/processed")
    out_path = Path("example_scalogram.png")

    # Load windows
    X = np.load(data_dir / "X_windows.npy")  # [N, T]

    if X.shape[0] == 0:
        raise RuntimeError("No windows found in X_windows.npy")

    # Take one 10s window (first one)
    w = X[0]

    # Compute scalogram used by Wavelet+CNN
    fs = 250
    n_scales = 64
    f_min, f_max = 0.5, 40.0
    S = window_to_scalogram_cwt(w, fs=fs, n_scales=n_scales, f_min=f_min, f_max=f_max)

    # Time and frequency axes for plotting
    T = w.shape[0]
    t = np.linspace(0, 10, T)  # 10 s window
    freqs = np.linspace(f_min, f_max, n_scales)

    plt.figure(figsize=(8, 4))
    # imshow expects [rows, cols] = [freq, time]
    plt.imshow(S, aspect="auto", origin="lower",
               extent=[t[0], t[-1], freqs[0], freqs[-1]], cmap="viridis")
    plt.colorbar(label="Normalized magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Example Scalogram (10 s ECG window)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved scalogram figure to {out_path.resolve()}")


if __name__ == "__main__":
    main()
