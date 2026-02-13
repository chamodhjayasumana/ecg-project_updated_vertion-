import json
import streamlit as st
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
from preprocess_utils import bandpass_notch, resample_to, z_norm
from window_features import extract_window_features
from personwise_windows import build_window_baseline, personwise_predict_windows
import joblib

RAW_DIR = Path("../data/raw/mitbih")
MODEL_PATH = Path("../models/cnn_bilstm.keras")
RF_MODEL_PATH = Path("../models/random_forest.pkl")

FS_TARGET = 250
WIN_SEC = 10
HOP_SEC = 5

st.set_page_config(layout="wide", page_title="ECG Interval Detection", page_icon="💓")
st.title("💓 Window-based Interval ECG Abnormality Detection (MIT-BIH)")

@st.cache_data
def ensure_mitbih():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not any(RAW_DIR.glob("*.hea")):
        wfdb.dl_database("mitdb", str(RAW_DIR))

def list_records():
    return sorted([p.stem for p in RAW_DIR.glob("*.hea")])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_rf_model():
    return joblib.load(RF_MODEL_PATH)

def make_windows(sig, fs):
    win = WIN_SEC * fs
    hop = HOP_SEC * fs
    windows, starts = [], []
    for start in range(0, len(sig) - win + 1, hop):
        end = start + win
        windows.append(z_norm(sig[start:end]))
        starts.append(start / fs)
    return np.stack(windows).astype(np.float32), np.array(starts, dtype=np.float32)

ensure_mitbih()
records = list_records()

if not MODEL_PATH.exists():
    st.error("Model not found. Run train_offline.py first to create models/cnn_bilstm.keras")
    st.stop()

rid = st.sidebar.selectbox("Choose record", records, index=0)
model = load_model()

if RF_MODEL_PATH.exists():
    rf_model = load_rf_model()
else:
    rf_model = None
    st.sidebar.warning("Random Forest model not found. Run train_random_forest.py to create it.")

# Load record
rec = wfdb.rdrecord(str(RAW_DIR / rid))
sig = rec.p_signal[:, 0].astype(np.float32)
fs_in = float(rec.fs)

# Preprocess + resample
sig = bandpass_notch(sig, fs_in)
sig, fs = resample_to(sig, fs_in, FS_TARGET)

Xw, starts = make_windows(sig, FS_TARGET)     # [N, T]
Xw_in = Xw[..., None]                         # [N, T, 1]

p = model.predict(Xw_in, verbose=0).ravel()
y_pred = (p >= 0.5).astype(int)

# Random Forest predictions on handcrafted features
if rf_model is not None:
    feats = np.array([extract_window_features(w, FS_TARGET) for w in Xw])
    rf_proba = rf_model.predict_proba(feats)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
else:
    rf_proba = None
    rf_pred = None

# Person-wise window baseline and predictions
pw = None
if len(Xw) >= 5:
    max_k = min(40, len(Xw))
    default_k = min(20, max_k)
    K = st.sidebar.slider("Baseline windows (K)", 5, max_k, default_k)
    baseline = build_window_baseline(Xw[:K], fs=FS_TARGET)
    pw = personwise_predict_windows(Xw, p, baseline, fs=FS_TARGET)

st.sidebar.metric("Windows", len(p))
st.sidebar.metric("CNN abnormal rate", f"{y_pred.mean()*100:.1f}%")
if rf_pred is not None:
    st.sidebar.metric("RF abnormal rate", f"{rf_pred.mean()*100:.1f}%")
if pw is not None:
    st.sidebar.metric("Person-wise abnormal %", f"{pw['abnormal'].mean()*100:.1f}%")

st.subheader("✅ CNN (BiLSTM) Prediction Summary")
st.metric("Total windows", len(p))
st.metric("CNN predicted abnormal windows", int(y_pred.sum()))
st.metric("CNN predicted abnormal rate", f"{y_pred.mean()*100:.1f}%")

if rf_pred is not None:
    st.subheader("🌲 Random Forest Prediction Summary")
    st.metric("RF predicted abnormal windows", int(rf_pred.sum()))
    st.metric("RF predicted abnormal rate", f"{rf_pred.mean()*100:.1f}%")

if pw is not None:
    st.subheader("🧍 Person-wise Window Analysis")
    st.metric("Deviation abnormal %", float(pw["abnormal"].mean() * 100.0))

# Plot signal sample
st.subheader("ECG Sample (filtered)")
fig = plt.figure(figsize=(12, 3))
plt.plot(sig[:FS_TARGET*20])
plt.title("First 20 seconds (Filtered + Resampled)")
st.pyplot(fig)

st.subheader("Window-wise Outputs (CNN vs RF vs Person-wise)")
timeline = []
for i in range(len(p)):
    row = {
        "start_s": float(starts[i]),
        "end_s": float(starts[i] + WIN_SEC),
        "p_cnn": float(p[i]),
        "cnn_label": "Abnormal" if y_pred[i] == 1 else "Normal",
    }

    if rf_proba is not None:
        row["p_rf"] = float(rf_proba[i])
        row["rf_label"] = "Abnormal" if rf_pred[i] == 1 else "Normal"
    else:
        row["p_rf"] = np.nan
        row["rf_label"] = "N/A"

    if pw is not None:
        row["z_dev"] = float(pw["z_score"][i])
        row["final_personwise"] = "Abnormal" if pw["abnormal"][i] == 1 else "Normal"
    else:
        row["z_dev"] = np.nan
        row["final_personwise"] = "N/A"

    timeline.append(row)

st.dataframe(timeline, use_container_width=True)

# Timeline plot
st.subheader("Abnormality Probability Timeline")
fig2 = plt.figure(figsize=(12, 3))
plt.plot(starts, p)
plt.axhline(0.5, linestyle="--")
plt.title("p(abnormal) per window")
plt.xlabel("time (s)")
plt.ylabel("probability")
st.pyplot(fig2)

st.subheader("🧪 Offline Model Results (Hold-out Test Set)")

RESULTS_DIR = Path("../results")
cm_path = RESULTS_DIR / "confusion_matrix.npy"
rep_json_path = RESULTS_DIR / "classification_report.json"
rep_csv_path = RESULTS_DIR / "classification_report.csv"

if cm_path.exists() and rep_json_path.exists():
    cm = np.load(cm_path)
    report_dict = json.loads(rep_json_path.read_text())

    # --- Confusion Matrix Plot ---
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "abnormal"])
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix (Offline Test Set)")
    st.pyplot(fig_cm)

    # --- Report table ---
    st.subheader("📋 Classification Report (Table)")
    df_report = pd.DataFrame(report_dict).transpose()
    st.dataframe(df_report, use_container_width=True)

    # --- Download buttons ---
    st.subheader("⬇ Download Results")

    # confusion matrix CSV
    cm_csv = pd.DataFrame(cm, index=["true_normal", "true_abnormal"], columns=["pred_normal", "pred_abnormal"]).to_csv()
    st.download_button(
        "Download Confusion Matrix (CSV)",
        data=cm_csv,
        file_name="confusion_matrix.csv",
        mime="text/csv"
    )

    # report JSON
    st.download_button(
        "Download Classification Report (JSON)",
        data=json.dumps(report_dict, indent=2),
        file_name="classification_report.json",
        mime="application/json"
    )

    # report CSV (if exists)
    if rep_csv_path.exists():
        st.download_button(
            "Download Classification Report (CSV)",
            data=rep_csv_path.read_text(),
            file_name="classification_report.csv",
            mime="text/csv"
        )

else:
    st.warning("No saved offline results found. Run train_offline.py to generate them.")
   