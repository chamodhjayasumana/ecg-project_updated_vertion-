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
from wavelet_utils import window_to_scalogram_cwt
import joblib
from personwise_logic import build_baseline_profile, personwise_predict_windows


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
rid = st.sidebar.selectbox("Choose record", records, index=0)
model_choice = st.sidebar.selectbox("Model", ["CNN-BiLSTM (raw)", "Wavelet+CNN"], index=0)

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

if model_choice == "Wavelet+CNN":
    wcnn_path = Path("../models/wavelet_cnn.keras")
    if not wcnn_path.exists():
        st.error("Wavelet+CNN model not found. Run train_wavelet_cnn.py first.")
        st.stop()

    wcnn = tf.keras.models.load_model(wcnn_path)

    # Build scalograms for current record windows
    S_list = [window_to_scalogram_cwt(w, fs=FS_TARGET, n_scales=64) for w in Xw]
    S = np.stack(S_list).astype(np.float32)[..., None]   # [N, scales, T, 1]

    p = wcnn.predict(S, verbose=0).ravel()
else:
    if not MODEL_PATH.exists():
        st.error("Model not found. Run train_offline.py first to create models/cnn_bilstm.keras")
        st.stop()

    model = load_model()
    p = model.predict(Xw_in, verbose=0).ravel()

y_pred = (p >= 0.5).astype(int)

# Person-wise baseline from "most normal" windows (lowest model abnormal probabilities)
pw = None
if len(Xw) >= 5:
    st.sidebar.markdown("### Person-wise baseline")
    enable_personwise = st.sidebar.checkbox("Enable person-wise", value=True)
    K_max = min(50, len(Xw))
    K = st.sidebar.slider("K baseline windows", 5, K_max, 20)
    dev_thr = st.sidebar.slider("Hard deviation threshold", 1.0, 20.0, 5.0, 0.5)
    risk_thr = st.sidebar.slider("Risk threshold (R)", 0.0, 1.0, 0.60, 0.01)
    p_thr = st.sidebar.slider("Model prob threshold (p)", 0.0, 1.0, 0.70, 0.01)

    if enable_personwise:
        # Choose baseline windows = lowest model abnormal probabilities
        idx_sorted = np.argsort(p)              # low p = normal
        base_idx = idx_sorted[:K]
        baseline_windows = [Xw[i] for i in base_idx]

        baseline = build_baseline_profile(
            person_id=f"rec_{rid}",
            windows=baseline_windows,
            fs=FS_TARGET,
            use_wavelet_energy=True
        )

        pw = personwise_predict_windows(
            windows=[w for w in Xw],
            model_probs=p,
            baseline=baseline,
            fs=FS_TARGET,
            risk_thr=float(risk_thr),
            p_thr=float(p_thr),
            td=3.0,
            hard_dev_thr=float(dev_thr),
            lam=0.6,
            smooth_min_run=3,
            consec_local=3,
            global_pct=0.20,
            use_wavelet_energy=True
        )

        st.subheader("🧍 Person-wise Window Results")
        st.write("Flags:", pw.flags)
        st.metric("Abnormal window rate (person-wise)", f"{pw.y_abn.mean()*100:.1f}%")

if pw is not None:
    timeline_pw = []
    for i in range(len(Xw)):
        timeline_pw.append({
            "start_s": float(starts[i]),
            "end_s": float(starts[i] + WIN_SEC),
            "p_model": float(p[i]),
            "dev_score": float(pw.d_score[i]),
            "risk": float(pw.risk[i]),
            "final_label": "Abnormal" if int(pw.y_abn[i]) == 1 else "Normal",
        })
    df_timeline_pw = pd.DataFrame(timeline_pw)
    st.dataframe(df_timeline_pw, use_container_width=True)
    csv_timeline_pw = df_timeline_pw.to_csv(index=False)
    st.download_button(
        "Download Person-wise Window Results (CSV)",
        data=csv_timeline_pw,
        file_name="personwise_window_results.csv",
        mime="text/csv"
    )

# Random Forest predictions on handcrafted features
if rf_model is not None:
    feats = np.array([extract_window_features(w, FS_TARGET) for w in Xw])
    rf_proba = rf_model.predict_proba(feats)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
else:
    rf_proba = None
    rf_pred = None

st.sidebar.metric("Windows", len(p))
st.sidebar.metric("CNN abnormal rate", f"{y_pred.mean()*100:.1f}%")
if rf_pred is not None:
    st.sidebar.metric("RF abnormal rate", f"{rf_pred.mean()*100:.1f}%")
if pw is not None:
    st.sidebar.metric("Person-wise abnormal %", f"{pw.y_abn.mean()*100:.1f}%")

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
    st.metric("Person-wise abnormal %", float(pw.y_abn.mean() * 100.0))

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
        row["z_dev"] = float(pw.d_score[i])
        row["final_personwise"] = "Abnormal" if int(pw.y_abn[i]) == 1 else "Normal"
    else:
        row["z_dev"] = np.nan
        row["final_personwise"] = "N/A"

    timeline.append(row)

df_timeline = pd.DataFrame(timeline)
st.dataframe(df_timeline, use_container_width=True)
csv_timeline = df_timeline.to_csv(index=False)
st.download_button(
    "Download Window-wise Outputs (CSV)",
    data=csv_timeline,
    file_name="window_wise_outputs.csv",
    mime="text/csv"
)

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

st.subheader("🧪 Wavelet+CNN Offline Results")

cm_path = Path("../results/wcnn_confusion.npy")
rp_path = Path("../results/wcnn_report.json")

if cm_path.exists() and rp_path.exists():
    cm = np.load(cm_path)
    rep = json.loads(rp_path.read_text())

    fig_cm, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(cm, display_labels=["normal","abnormal"]).plot(ax=ax, values_format="d")
    ax.set_title("Wavelet+CNN Confusion Matrix")
    st.pyplot(fig_cm)

    st.json(rep)
else:
    st.info("Run train_wavelet_cnn.py to generate Wavelet+CNN results.")

st.subheader("🌲 Random Forest Baseline (Offline Test Set)")

rf_cm_path = Path("../results/rf_confusion.npy")
rf_rep_path = Path("../results/rf_report.json")

if rf_cm_path.exists() and rf_rep_path.exists():
    rf_cm = np.load(rf_cm_path)
    rf_rep = json.loads(rf_rep_path.read_text())
    st.write("RF Confusion Matrix:", rf_cm)
    st.json(rf_rep)
else:
    st.warning("Run train_random_forest.py to generate RF baseline results.")
   