import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from config import CSV_PATH, MODEL_SAVE_PATH, SCALER_PATH, GENRES, EPOCHS

# ── Load CSV ──────────────────────────────────────────────────────────────────
df    = pd.read_csv(CSV_PATH)
X     = df.drop(columns=["filename", "label"]).values.astype(np.float32)
y_raw = df["label"].values

# ── Encode labels to one-hot vectors ─────────────────────────────────────────
genre_to_index = {genre: i for i, genre in enumerate(GENRES)}
y = to_categorical([genre_to_index[g] for g in y_raw], num_classes=len(GENRES))

# ── Normalize features ────────────────────────────────────────────────────────
scaler = StandardScaler()
X      = scaler.fit_transform(X)

# ── Save scaler ───────────────────────────────────────────────────────────────
os.makedirs("saved_model", exist_ok=True)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# ── Train / val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Build MLP ─────────────────────────────────────────────────────────────────
inputs  = Input(shape=(X_train.shape[1],))
x       = Dense(128, activation="relu")(inputs)
x       = Dropout(0.3)(x) 
x       = Dense(64, activation="relu")(x)
x       = Dropout(0.3)(x) 
outputs = Dense(len(GENRES), activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ── Train ─────────────────────────────────────────────────────────────────────
# step2_train.py — add this callback

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    verbose=1
)

# ── Save model ────────────────────────────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
# ── Plot curves ───────────────────────────────────────────────────────────────
def plot_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], linestyle="--", label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], linestyle="--", label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.savefig("training_curves.png")
    plt.show()

plot_curves(history)