import numpy as np
from window_features import extract_window_features

def build_window_baseline(windows, fs=250):
    feats = np.array([extract_window_features(w, fs) for w in windows])
    return {
        "mean": feats.mean(axis=0),
        "std": feats.std(axis=0) + 1e-6
    }

def personwise_predict_windows(windows, model_probs, baseline, fs=250, dev_thr=3.0):
    feats = np.array([extract_window_features(w, fs) for w in windows])
    z = np.linalg.norm((feats - baseline["mean"]) / baseline["std"], axis=1)

    abnormal = ((model_probs > 0.5) | (z > dev_thr)).astype(int)

    return {
        "z_score": z,
        "abnormal": abnormal
    }
