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

RAW_DIR = Path("../data/raw/mitbih")
MODEL_PATH = Path("../models/cnn_bilstm.keras")

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

st.sidebar.metric("Windows", len(p))
st.sidebar.metric("Abnormal rate", f"{y_pred.mean()*100:.1f}%")

st.subheader("✅ Prediction Summary (Selected Record)")
st.metric("Total windows", len(p))
st.metric("Predicted abnormal windows", int(y_pred.sum()))
st.metric("Predicted abnormal rate", f"{y_pred.mean()*100:.1f}%")

# Plot signal sample
st.subheader("ECG Sample (filtered)")
fig = plt.figure(figsize=(12, 3))
plt.plot(sig[:FS_TARGET*20])
plt.title("First 20 seconds (Filtered + Resampled)")
st.pyplot(fig)

# Timeline table
st.subheader("Interval Timeline (10s windows, 5s overlap)")
timeline = []
for i in range(len(p)):
    timeline.append({
        "start_s": float(starts[i]),
        "end_s": float(starts[i] + WIN_SEC),
        "p_abnormal": float(p[i]),
        "label": "Abnormal" if y_pred[i] == 1 else "Normal"
    })
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
   