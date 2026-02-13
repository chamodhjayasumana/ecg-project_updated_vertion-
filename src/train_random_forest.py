import numpy as np
from pathlib import Path
import joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from personwise_logic import extract_features_window

DATA = Path("../data/processed")
MODELS = Path("../models")
RESULTS = Path("../results")
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

FS = 250

# Load window dataset
X = np.load(DATA / "X_windows.npy")                       # [N, T]
y = np.load(DATA / "y_windows.npy")                       # [N]
rec_ids = np.load(DATA / "rec_ids.npy", allow_pickle=True)

# Build feature matrix (same features you use for person-wise)
# Keep feature order consistent
feat_names = None
F = []
for w in X:
    feats = extract_features_window(w, fs=FS, rpeaks=None, use_wavelet_energy=True)
    # On the first window, choose only features that are finite
    if feat_names is None:
        all_keys = sorted(feats.keys())
        feat_names = [k for k in all_keys if np.isfinite(feats[k])]
    F.append([feats[k] for k in feat_names])
F = np.array(F, dtype=np.float32)

# Remove any windows with NaN or infinite feature values
finite_mask = np.isfinite(F).all(axis=1)
if not finite_mask.all():
    dropped = np.size(finite_mask) - np.count_nonzero(finite_mask)
    print(f"Dropping {dropped} windows due to non-finite feature values (NaN/Inf).")
    F = F[finite_mask]
    y = y[finite_mask]
    rec_ids = rec_ids[finite_mask]

# Record-wise split (no leakage)
unique = np.unique(rec_ids)
np.random.seed(42)
np.random.shuffle(unique)
test_ids = set(unique[:max(5, int(0.2*len(unique)))])

# Boolean masks for record-wise split
test_mask = np.array([r in test_ids for r in rec_ids], dtype=bool)
train_mask = np.logical_not(test_mask)

Xtr, Xte = F[train_mask], F[test_mask]
ytr, yte = y[train_mask], y[test_mask]

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(Xtr, ytr)

yp = rf.predict(Xte)
cm = confusion_matrix(yte, yp)
rep = classification_report(yte, yp, target_names=["normal","abnormal"], output_dict=True, zero_division=0)

# Save model + metadata + results
joblib.dump({"model": rf, "feat_names": feat_names}, MODELS / "random_forest.pkl")
np.save(RESULTS / "rf_confusion.npy", cm)
json.dump(rep, open(RESULTS / "rf_report.json", "w"), indent=2)

print("RF Confusion:\n", cm)
print("Saved RF model + results.")
