# personwise_logic.py
# ---------------------------------------------
# Person-wise baseline + interval prediction logic
# Ready to plug into a Streamlit ECG pipeline.
#
# Assumptions:
# - You already have windows (numpy arrays) from your ECG signal.
# - You already have a trained model that can output probability p(abnormal) per window.
#
# You can start with simple features (RR/HRV optional if you have R-peaks).
# This code is written to work even WITHOUT R-peak detection (it will still run).

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# 1) Feature extraction
# -----------------------------
def _safe_std(x: np.ndarray, eps: float = 1e-8) -> float:
    s = float(np.std(x))
    return s if s > eps else eps


def extract_features_window(
    w: np.ndarray,
    fs: int,
    rpeaks: Optional[np.ndarray] = None,
    use_wavelet_energy: bool = False,
) -> Dict[str, float]:
    """
    Extract simple, robust features for person-wise baseline comparison.

    Parameters
    ----------
    w : np.ndarray
        ECG window (shape: [n_samples] or [n_samples, n_channels]).
        If multi-lead, pass a single lead or flatten/choose lead before calling.
    fs : int
        Sampling rate.
    rpeaks : Optional[np.ndarray]
        R-peak indices within the *full signal* or within the window.
        If you pass window-local rpeaks, ensure they are indices in [0, len(w)).
        If None, RR/HRV features are skipped.
    use_wavelet_energy : bool
        If True, attempts to compute crude band energies using FFT bands (no pywt needed).
        (You can replace this with true wavelet energies if you use pywt.)

    Returns
    -------
    Dict[str, float]
    """
    # If window is 2D (multi-lead), take lead-0 by default
    if w.ndim == 2:
        w = w[:, 0]

    w = np.asarray(w, dtype=np.float32)

    feats: Dict[str, float] = {}
    feats["mean"] = float(np.mean(w))
    feats["std"] = float(np.std(w))
    feats["rms"] = float(np.sqrt(np.mean(w**2)))
    feats["ptp"] = float(np.ptp(w))  # peak-to-peak amplitude
    feats["abs_mean"] = float(np.mean(np.abs(w)))
    feats["abs_max"] = float(np.max(np.abs(w)))
    feats["skew_approx"] = float(np.mean(((w - feats["mean"]) / _safe_std(w)) ** 3))
    feats["kurt_approx"] = float(np.mean(((w - feats["mean"]) / _safe_std(w)) ** 4))

    # Optional: RR / HRV features if R-peaks are available
    # rpeaks should be window-local indices for correct RR calc in this window.
    if rpeaks is not None and len(rpeaks) >= 2:
        rpeaks = np.asarray(rpeaks, dtype=int)
        rr = np.diff(rpeaks) / float(fs)  # seconds
        feats["rr_mean"] = float(np.mean(rr))
        feats["rr_std"] = float(np.std(rr))
        # Common HRV measures (rough, window-based)
        feats["rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) >= 2 else 0.0
        feats["hr_mean"] = float(60.0 / (np.mean(rr) + 1e-8))
    else:
        # Keep keys consistent (optional)
        feats["rr_mean"] = np.nan
        feats["rr_std"] = np.nan
        feats["rmssd"] = np.nan
        feats["hr_mean"] = np.nan

    # Optional crude frequency band energies (acts like a lightweight proxy)
    if use_wavelet_energy:
        # FFT power spectrum
        x = w - np.mean(w)
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

        def band_energy(f_lo: float, f_hi: float) -> float:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            return float(np.sum(spec[mask])) if np.any(mask) else 0.0

        feats["e_0_5_3"] = band_energy(0.5, 3.0)
        feats["e_3_10"] = band_energy(3.0, 10.0)
        feats["e_10_20"] = band_energy(10.0, 20.0)
        feats["e_20_40"] = band_energy(20.0, 40.0)

    return feats


# -----------------------------
# 2) Person baseline profile
# -----------------------------
@dataclass
class BaselineProfile:
    person_id: str
    fs: int
    feature_names: List[str]
    mu: Dict[str, float]
    sigma: Dict[str, float]
    # Optional metadata
    n_windows: int

    def to_dict(self) -> Dict:
        return asdict(self)


def build_baseline_profile(
    person_id: str,
    windows: List[np.ndarray],
    fs: int,
    rpeaks_per_window: Optional[List[Optional[np.ndarray]]] = None,
    use_wavelet_energy: bool = False,
) -> BaselineProfile:
    """
    Build baseline stats (mean + std) per feature from "normal-for-this-person" windows.
    """
    if rpeaks_per_window is None:
        rpeaks_per_window = [None] * len(windows)

    feats_list: List[Dict[str, float]] = []
    for w, rp in zip(windows, rpeaks_per_window):
        feats_list.append(extract_features_window(w, fs=fs, rpeaks=rp, use_wavelet_energy=use_wavelet_energy))

    # Convert to matrix
    feature_names = sorted(feats_list[0].keys())
    X = np.array([[f.get(k, np.nan) for k in feature_names] for f in feats_list], dtype=np.float32)

    mu: Dict[str, float] = {}
    sigma: Dict[str, float] = {}
    for j, k in enumerate(feature_names):
        col = X[:, j]
        # Ignore NaNs in baseline stats
        mu_k = float(np.nanmean(col))
        sig_k = float(np.nanstd(col))
        mu[k] = mu_k
        sigma[k] = sig_k if sig_k > 1e-6 else 1e-6

    return BaselineProfile(
        person_id=person_id,
        fs=fs,
        feature_names=feature_names,
        mu=mu,
        sigma=sigma,
        n_windows=len(windows),
    )


# -----------------------------
# 3) Deviation scoring + fusion
# -----------------------------
def deviation_score_z(
    feats: Dict[str, float],
    baseline: BaselineProfile,
    weights: Optional[Dict[str, float]] = None,
    mode: str = "weighted_sum",  # "weighted_sum" or "max"
) -> float:
    """
    Compute deviation score D(w) from baseline using z-scores.
    """
    zs: List[float] = []
    for k in baseline.feature_names:
        x = feats.get(k, np.nan)
        if np.isnan(x):
            continue
        z = (x - baseline.mu[k]) / (baseline.sigma[k] + 1e-8)
        z_abs = float(abs(z))
        if weights and k in weights:
            z_abs *= float(weights[k])
        zs.append(z_abs)

    if not zs:
        return 0.0

    if mode == "max":
        return float(np.max(zs))
    return float(np.sum(zs))


def fuse_risk(
    p_model: float,
    d_score: float,
    td: float = 3.0,        # deviation normalization threshold
    lam: float = 0.6,       # weight for model probability
) -> float:
    """
    Fuse model probability and normalized deviation into final risk score R(w).
    """
    p_model = float(np.clip(p_model, 0.0, 1.0))
    d_norm = float(np.clip(d_score / td, 0.0, 1.0))
    return lam * p_model + (1.0 - lam) * d_norm


# -----------------------------
# 4) Interval rules / smoothing
# -----------------------------
def smooth_binary_predictions(
    y: np.ndarray,
    min_run: int = 3,
) -> np.ndarray:
    """
    Remove isolated spikes: keep abnormal only if it appears in runs >= min_run.
    """
    y = y.astype(int).copy()
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            run_len = j - i
            if run_len < min_run:
                y[i:j] = 0
            i = j
        else:
            i += 1
    return y


def compute_interval_flags(
    y_abn: np.ndarray,
    consec_local: int = 3,
    global_pct: float = 0.20,
) -> Dict[str, bool]:
    """
    High-level interval decisions: localized abnormality and global risk.
    """
    y_abn = y_abn.astype(int)
    localized = bool(np.any(np.convolve(y_abn, np.ones(consec_local, dtype=int), mode="valid") >= consec_local))
    global_risk = bool(np.mean(y_abn) >= global_pct)
    return {"localized_abnormality": localized, "global_risk": global_risk}


# -----------------------------
# 5) Main person-wise inference
# -----------------------------
@dataclass
class PersonWiseResult:
    p_model: np.ndarray        # per-window model probability
    d_score: np.ndarray        # per-window deviation score
    risk: np.ndarray           # fused per-window risk
    y_abn: np.ndarray          # final abnormal labels (0/1) after rules
    flags: Dict[str, bool]     # localized/global etc.


def personwise_predict_windows(
    windows: List[np.ndarray],
    model_probs: np.ndarray,
    baseline: BaselineProfile,
    fs: int,
    rpeaks_per_window: Optional[List[Optional[np.ndarray]]] = None,
    # thresholds
    risk_thr: float = 0.60,
    p_thr: float = 0.70,
    td: float = 3.0,
    hard_dev_thr: float = 5.0,
    # fusion
    lam: float = 0.6,
    # deviation
    weights: Optional[Dict[str, float]] = None,
    dev_mode: str = "weighted_sum",
    # smoothing & interval rules
    smooth_min_run: int = 3,
    consec_local: int = 3,
    global_pct: float = 0.20,
    use_wavelet_energy: bool = False,
) -> PersonWiseResult:
    """
    Produce person-wise interval predictions from:
    - windows
    - model_probs (p abnormal per window)
    - baseline profile
    """

    if rpeaks_per_window is None:
        rpeaks_per_window = [None] * len(windows)

    model_probs = np.asarray(model_probs, dtype=np.float32)
    assert len(model_probs) == len(windows), "model_probs length must match windows"

    d_scores = np.zeros(len(windows), dtype=np.float32)
    risks = np.zeros(len(windows), dtype=np.float32)
    y = np.zeros(len(windows), dtype=np.int32)

    for i, (w, p, rp) in enumerate(zip(windows, model_probs, rpeaks_per_window)):
        feats = extract_features_window(w, fs=fs, rpeaks=rp, use_wavelet_energy=use_wavelet_energy)
        d = deviation_score_z(feats, baseline, weights=weights, mode=dev_mode)
        r = fuse_risk(p_model=float(p), d_score=float(d), td=td, lam=lam)

        d_scores[i] = d
        risks[i] = r

        # Window abnormal decision logic
        is_abn = (r >= risk_thr) or (p >= p_thr) or (d >= hard_dev_thr)
        y[i] = 1 if is_abn else 0

    # Smooth spikes (optional)
    y_smooth = smooth_binary_predictions(y, min_run=smooth_min_run)

    # Interval flags
    flags = compute_interval_flags(y_smooth, consec_local=consec_local, global_pct=global_pct)

    return PersonWiseResult(
        p_model=model_probs,
        d_score=d_scores,
        risk=risks,
        y_abn=y_smooth,
        flags=flags,
    )
