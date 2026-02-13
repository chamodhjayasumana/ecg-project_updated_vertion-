import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import json

DATA = Path("../data/processed/wavelet")
MODELS = Path("../models")
RESULTS = Path("../results")
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

def build_wavelet_cnn(input_shape):
    # input_shape = (scales, T, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

S = np.load(DATA / "S.npy")                      # [N, scales, T, 1]
y = np.load(DATA / "y.npy")                      # [N]
rec_ids = np.load(DATA / "rec_ids.npy", allow_pickle=True)

# Record-wise split (no leakage)
unique = np.unique(rec_ids)
np.random.seed(42)
np.random.shuffle(unique)
n_test = max(5, int(0.2 * len(unique)))
test_ids = set(unique[:n_test])

test_mask = np.array([r in test_ids for r in rec_ids])
train_mask = ~test_mask

X_train, y_train = S[train_mask], y[train_mask]
X_test, y_test = S[test_mask], y[test_mask]

print("Train:", X_train.shape, "Test:", X_test.shape)

model = build_wavelet_cnn(X_train.shape[1:])

cb = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(str(MODELS / "wavelet_cnn.keras"), save_best_only=True, monitor="val_loss")
]

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=32,
    callbacks=cb,
    verbose=1
)

p = model.predict(X_test).ravel()
y_pred = (p >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred, target_names=["normal","abnormal"], output_dict=True, zero_division=0)

np.save(RESULTS / "wcnn_confusion.npy", cm)
json.dump(rep, open(RESULTS / "wcnn_report.json", "w"), indent=2)

print("Wavelet+CNN Confusion:\n", cm)
print("Saved model -> models/wavelet_cnn.keras")
