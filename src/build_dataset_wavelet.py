import numpy as np
from pathlib import Path
from wavelet_utils import window_to_scalogram_cwt

DATA = Path("../data/processed")
OUT = Path("../data/processed/wavelet")
OUT.mkdir(parents=True, exist_ok=True)

FS = 250

X = np.load(DATA / "X_windows.npy")            # [N, T]
y = np.load(DATA / "y_windows.npy")            # [N]
rec_ids = np.load(DATA / "rec_ids.npy", allow_pickle=True)

print("Building scalograms...")
S_list = []
for i, w in enumerate(X):
    S = window_to_scalogram_cwt(w, fs=FS, n_scales=64, f_min=0.5, f_max=40.0)
    S_list.append(S)

S_all = np.stack(S_list).astype(np.float32)     # [N, scales, T]
S_all = S_all[..., None]                        # [N, scales, T, 1] channel

np.save(OUT / "S.npy", S_all)
np.save(OUT / "y.npy", y)
np.save(OUT / "rec_ids.npy", rec_ids)

print("Saved:", S_all.shape, "to", (OUT / "S.npy").resolve())
