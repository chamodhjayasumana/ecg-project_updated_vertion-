import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Dense, Bidirectional, LSTM

DATA_DIR = Path("../data/processed")
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def build_model(input_shape):
    model = Sequential([
        Conv1D(64, 11, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 7, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    X = np.load(DATA_DIR / "X_windows.npy")         # [N, T]
    y = np.load(DATA_DIR / "y_windows.npy")         # [N]
    rec_ids = np.load(DATA_DIR / "rec_ids.npy", allow_pickle=True)

    # reshape for Conv1D: [N, T, 1]
    X = X[..., None].astype(np.float32)

    # record-wise split
    unique_recs = np.unique(rec_ids)
    np.random.seed(42)
    np.random.shuffle(unique_recs)

    n_test = max(5, int(0.2 * len(unique_recs)))
    test_recs = set(unique_recs[:n_test])
    train_recs = set(unique_recs[n_test:])

    train_mask = np.array([r in train_recs for r in rec_ids])
    test_mask  = np.array([r in test_recs for r in rec_ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("Train windows:", X_train.shape, "Test windows:", X_test.shape)
    print("Train abnormal%:", y_train.mean()*100, "Test abnormal%:", y_test.mean()*100)

    model = build_model((X_train.shape[1], 1))

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "cnn_bilstm.keras"), save_best_only=True, monitor="val_loss")
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=64,
        callbacks=cb,
        verbose=1
    )

    # Evaluate
    p = model.predict(X_test).ravel()
    y_pred = (p >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["normal", "abnormal"],
        output_dict=True,
        zero_division=0,
    )

    # Save confusion matrix + report
    np.save(RESULTS_DIR / "confusion_matrix.npy", cm)
    with open(RESULTS_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nConfusion Matrix:\n", cm)
    print("\nReport:\n", json.dumps(report, indent=2))
    print("\nSaved to:", RESULTS_DIR.resolve())

if __name__ == "__main__":
    main()
