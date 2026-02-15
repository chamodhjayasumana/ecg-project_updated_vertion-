import json
import pandas as pd
from pathlib import Path

RESULTS = Path("../results")

# Map human-readable model names to the actual report files
# produced by the corresponding training scripts.
MODELS = {
    "CNN-BiLSTM": "classification_report.json",  # from train_offline.py
    "Wavelet+CNN": "wcnn_report.json",           # from train_wavelet_cnn.py
    "Random Forest": "rf_report.json",           # from train_random_forest.py
}

def load_report(path):
    with open(path) as f:
        return json.load(f)

def extract_metrics(rep):
    # assumes binary classification: abnormal class = "1" or "abnormal"
    if "abnormal" in rep:
        cls = rep["abnormal"]
    elif "1" in rep:
        cls = rep["1"]
    else:
        cls = list(rep.values())[0]

    return {
        "Accuracy": rep["accuracy"],
        "Precision": cls["precision"],
        "Recall": cls["recall"],
        "F1": cls["f1-score"]
    }

def build_comparison_table():
    rows = []

    for model, file in MODELS.items():
        p = RESULTS / file
        if not p.exists():
            continue

        rep = load_report(p)
        m = extract_metrics(rep)
        m["Model"] = model
        rows.append(m)

    df = pd.DataFrame(rows)
    return df[["Model","Accuracy","Precision","Recall","F1"]]
