import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from window_features import extract_window_features
import joblib
import json

DATA = Path("../data/processed")
RESULTS = Path("../results")
MODELS = Path("../models")

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

X = np.load(DATA / "X_windows.npy")
y = np.load(DATA / "y_windows.npy")
rec_ids = np.load(DATA / "rec_ids.npy", allow_pickle=True)

fs = 250

# Feature extraction
print("Extracting handcrafted features...")
F = np.array([extract_window_features(w, fs) for w in X])

# Record-wise split
unique = np.unique(rec_ids)
np.random.shuffle(unique)
test_ids = set(unique[:5])

mask_test = np.array([r in test_ids for r in rec_ids])
mask_train = ~mask_test

Xtr, Xte = F[mask_train], F[mask_test]
ytr, yte = y[mask_train], y[mask_test]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(Xtr, ytr)

yp = rf.predict(Xte)

cm = confusion_matrix(yte, yp)
rep = classification_report(yte, yp, target_names=["normal", "abnormal"], output_dict=True)

joblib.dump(rf, MODELS / "random_forest.pkl")
np.save(RESULTS / "rf_confusion.npy", cm)
json.dump(rep, open(RESULTS / "rf_report.json", "w"), indent=2)

print("RF Confusion:\n", cm)
print("RF done.")
