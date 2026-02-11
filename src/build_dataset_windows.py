import json
import numpy as np
import wfdb
from pathlib import Path
from preprocess_utils import bandpass_notch, resample_to, z_norm

RAW_DIR = Path("../data/raw/mitbih")
OUT_DIR = Path("../data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS_TARGET = 250
WIN_SEC = 10
HOP_SEC = 5

# Normal-like symbols in MIT-BIH annotations
NORMAL = {"N", "L", "R", "e", "j"}

def label_window(ann_samples, ann_symbols, start, end):
    """Binary label: 0 normal, 1 abnormal.
       abnormal if ANY beat symbol in window is not normal-like.
    """
    mask = (ann_samples >= start) & (ann_samples < end)
    if not np.any(mask):
        # no annotations inside window => treat as normal (or skip; your choice)
        return 0
    sym = ann_symbols[mask]
    for s in sym:
        if s not in NORMAL:
            return 1
    return 0

def get_record_ids():
    # safest: use wfdb list
    # but offline: after download, use *.hea
    return sorted([p.stem for p in RAW_DIR.glob("*.hea")])

def main():
    # Download whole mitdb once (first run)
    if not any(RAW_DIR.glob("*.hea")):
        print("Downloading MIT-BIH (mitdb) ...")
        wfdb.dl_database("mitdb", str(RAW_DIR))

    record_ids = get_record_ids()
    print("Records found:", len(record_ids))

    X, y, rec_ids, starts = [], [], [], []

    for rid in record_ids:
        try:
            base = str(RAW_DIR / rid)
            rec = wfdb.rdrecord(base)
            ann = wfdb.rdann(base, "atr")

            sig = rec.p_signal[:, 0].astype(np.float32)
            fs_in = float(rec.fs)

            sig = bandpass_notch(sig, fs_in)
            sig, fs = resample_to(sig, fs_in, FS_TARGET)

            # Map annotation samples to new fs
            ann_samples = (np.array(ann.sample) * (FS_TARGET / fs_in)).astype(int)
            ann_symbols = np.array(ann.symbol)

            win = WIN_SEC * FS_TARGET
            hop = HOP_SEC * FS_TARGET

            for start in range(0, len(sig) - win + 1, hop):
                end = start + win
                seg = z_norm(sig[start:end])
                lbl = label_window(ann_samples, ann_symbols, start, end)

                X.append(seg)
                y.append(lbl)
                rec_ids.append(rid)
                starts.append(start / FS_TARGET)

            print(f"{rid}: windows={(len(sig)-win)//hop + 1}")
        except Exception as e:
            print("Skip", rid, "->", e)

    X = np.stack(X).astype(np.float32)              # [N, T]
    y = np.array(y, dtype=np.int64)                # [N]
    rec_ids = np.array(rec_ids, dtype=object)
    starts = np.array(starts, dtype=np.float32)

    np.save(OUT_DIR / "X_windows.npy", X)
    np.save(OUT_DIR / "y_windows.npy", y)
    np.save(OUT_DIR / "rec_ids.npy", rec_ids)
    np.save(OUT_DIR / "starts.npy", starts)

    meta = {
        "fs_target": FS_TARGET,
        "win_sec": WIN_SEC,
        "hop_sec": HOP_SEC,
        "labeling": "binary: abnormal if any non-normal beat symbol appears in window",
        "normal_symbols": sorted(list(NORMAL)),
        "records_count": int(len(set(rec_ids.tolist()))),
        "windows_total": int(len(y)),
        "abnormal_rate": float(y.mean()),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved dataset to:", OUT_DIR.resolve())
    print(meta)

if __name__ == "__main__":
    main()
