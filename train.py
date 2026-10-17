"""
MoodMate - Step 1: train the emotion detection model.

What this script does
---------------------
1. Downloads the "Emotion" dataset (Saravia et al., 2018): 416,809 English
   sentences, each labelled with one of 6 emotions. The standard 2,000-sentence
   validation and test sets are kept separate (and removed from training data).
2. Trains two models so we can compare classical ML with deep learning:
     - Baseline   : TF-IDF + Logistic Regression       (machine learning)
     - Main model : Word Embedding + Bidirectional LSTM (deep learning / RNN)
3. Evaluates both on the unseen test set, plus a small "real-world check" of
   sentences we wrote ourselves, and saves charts + metrics.

Run:  python train.py          (about 5-10 minutes on a normal laptop CPU)
Outputs:  model/   -> the trained LSTM + its vocabulary (used by app.py)
          results/ -> accuracy, F1, confusion matrix, training curves
"""

import json
import os
import random

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info spam

import matplotlib
matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 42
VOCAB_SIZE = 20000      # keep the 20,000 most frequent words
MAX_LEN = 50            # every sentence is cut / padded to 50 words
EMBED_DIM = 128         # each word becomes a 128-number vector
LSTM_UNITS = 64
EPOCHS = 8
BATCH_SIZE = 256

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
DATA_URL = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/{}-00000-of-00001.parquet"

# Sentences we wrote ourselves (not from the dataset) to see how the model
# copes with everyday writing. Benchmarks can look better than real life!
REAL_WORLD_CHECK = {
    "sadness": ["i feel so lonely since my best friend moved away",
                "i failed my driving test again and i feel hopeless",
                "nobody remembered my birthday today",
                "i am not happy at all today"],
    "joy": ["i got the internship i applied for, i am so happy",
            "today was a wonderful day at the beach with friends",
            "i finally understood the topic i was struggling with and i feel great",
            "i passed all my exams and feel amazing"],
    "love": ["i love my little sister so much",
             "i feel so much affection for my grandparents",
             "i adore spending time with my best friend",
             "had dinner with my parents and i feel so loved and cared for"],
    "anger": ["i feel so angry that my brother broke my headphones",
              "i am furious that the shop refused to refund me",
              "people who cut in line make me so mad",
              "my boss yelled at me for no reason and i am irritated"],
    "fear": ["i am scared to walk home alone at night",
             "i feel anxious about my job interview tomorrow",
             "i am terrified of the dentist appointment",
             "i feel so stressed and worried about the presentation on monday"],
    "surprise": ["i was shocked to see snow in october",
                 "i could not believe it when they announced my name",
                 "i was amazed by how big the city was",
                 "wow i did not expect that ending at all"],
}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "model")
RESULTS_DIR = os.path.join(HERE, "results")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------------------
# 1. Load the data (downloaded once, then read from ./data)
# ----------------------------------------------------------------------------
def load_split(name, remote_path):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"Downloading {name} data...")
        os.makedirs(DATA_DIR, exist_ok=True)
        pd.read_parquet(DATA_URL.format(remote_path)).to_csv(path, index=False)
    return pd.read_csv(path)


full_df = load_split("full", "unsplit/train")
val_df = load_split("validation", "split/validation")
test_df = load_split("test", "split/test")

# The full dataset also contains the validation/test sentences - remove them so
# the model is always tested on sentences it has never seen. Also drop duplicates.
held_out = set(val_df["text"]) | set(test_df["text"])
train_df = full_df[~full_df["text"].isin(held_out)].drop_duplicates("text").reset_index(drop=True)

print(f"Train: {len(train_df)}  Validation: {len(val_df)}  Test: {len(test_df)}")
print("Class balance (train):")
print(train_df["label"].map(dict(enumerate(EMOTIONS))).value_counts(normalize=True).round(3).to_string(), "\n")

x_train, y_train = train_df["text"].values, train_df["label"].values
x_val, y_val = val_df["text"].values, val_df["label"].values
x_test, y_test = test_df["text"].values, test_df["label"].values
x_real = [s for e in EMOTIONS for s in REAL_WORLD_CHECK[e]]
y_real = np.array([i for i, e in enumerate(EMOTIONS) for _ in REAL_WORLD_CHECK[e]])


# ----------------------------------------------------------------------------
# 2. Baseline: TF-IDF + Logistic Regression
# ----------------------------------------------------------------------------
print("Training baseline (TF-IDF + Logistic Regression)...")
tfidf = TfidfVectorizer(max_features=VOCAB_SIZE, ngram_range=(1, 2))
baseline = LogisticRegression(max_iter=1000)
baseline.fit(tfidf.fit_transform(x_train), y_train)
base_pred = baseline.predict(tfidf.transform(x_test))
base_acc = accuracy_score(y_test, base_pred)
base_f1 = f1_score(y_test, base_pred, average="macro")
base_real = np.mean(baseline.predict(tfidf.transform(x_real)) == y_real)
print(f"Baseline  -> accuracy {base_acc:.3f}  macro-F1 {base_f1:.3f}  real-world {base_real:.0%}\n")


# ----------------------------------------------------------------------------
# 3. Main model: Bidirectional LSTM (a type of RNN)
# ----------------------------------------------------------------------------
# Text -> list of word IDs. "i feel great" -> [2, 3, 412, 0, 0, ...]
vectorizer = layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN)
vectorizer.adapt(x_train)
vocab = vectorizer.get_vocabulary()

def encode(texts):
    return vectorizer(tf.constant(list(texts))).numpy()

seq_train, seq_val, seq_test, seq_real = encode(x_train), encode(x_val), encode(x_test), encode(x_real)
one_hot = lambda y: keras.utils.to_categorical(y, len(EMOTIONS))

model = keras.Sequential([
    keras.Input(shape=(MAX_LEN,)),
    # Embedding: learns a meaning-vector for every word ("happy" ends up near "glad")
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
    layers.SpatialDropout1D(0.2),             # randomly hide whole words -> less memorising
    # Bidirectional LSTM: reads the sentence left->right AND right->left,
    # so it can use context on both sides of every word
    layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
    layers.Dropout(0.5),                      # randomly switch off neurons -> less overfitting
    layers.Dense(64, activation="relu"),
    layers.Dense(len(EMOTIONS), activation="softmax"),  # one probability per emotion
])
# Label smoothing: train towards 90% (not 100%) certainty, so the app's
# confidence scores are more honest and the model memorises less.
model.compile(optimizer="adam",
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=["accuracy"])
model.summary()

print("\nTraining Bidirectional LSTM...")
history = model.fit(
    seq_train, one_hot(y_train),
    validation_data=(seq_val, one_hot(y_val)),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)],
    verbose=2,
)

lstm_pred = model.predict(seq_test, verbose=0).argmax(axis=1)
lstm_acc = accuracy_score(y_test, lstm_pred)
lstm_f1 = f1_score(y_test, lstm_pred, average="macro")
real_pred = model.predict(seq_real, verbose=0).argmax(axis=1)
lstm_real = np.mean(real_pred == y_real)
print(f"\nBiLSTM    -> accuracy {lstm_acc:.3f}  macro-F1 {lstm_f1:.3f}  real-world {lstm_real:.0%}")


# ----------------------------------------------------------------------------
# 4. Save the model and results
# ----------------------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save an uncompiled copy (no optimizer state) -> smaller file for the app
export = keras.models.clone_model(model)
export.set_weights(model.get_weights())
export.save(os.path.join(MODEL_DIR, "moodmate_lstm.keras"))
with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
    json.dump({"vocab": vocab, "max_len": MAX_LEN, "emotions": EMOTIONS}, f)

report = classification_report(y_test, lstm_pred, target_names=EMOTIONS, digits=3)
real_lines = [f"{'OK ' if p == y else 'X  '} expected {EMOTIONS[y]:9s} got {EMOTIONS[p]:9s} | {s}"
              for s, y, p in zip(x_real, y_real, real_pred)]
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write("Bidirectional LSTM - test set (2,000 unseen sentences)\n\n" + report)
    f.write(f"\nReal-world check ({lstm_real:.0%} correct)\n\n" + "\n".join(real_lines) + "\n")
print("\n" + report)
print("\n".join(real_lines))

metrics = {
    "baseline_tfidf_logreg": {"accuracy": round(base_acc, 4), "macro_f1": round(base_f1, 4),
                              "real_world": round(float(base_real), 4)},
    "bilstm": {"accuracy": round(lstm_acc, 4), "macro_f1": round(lstm_f1, 4),
               "real_world": round(float(lstm_real), 4)},
    "epochs_trained": len(history.history["loss"]),
    "train_size": len(x_train), "test_size": len(x_test), "real_world_size": len(x_real),
}
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# --- Charts -----------------------------------------------------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 130,
})

# Training curves: one chart per measure (never two y-scales on one chart)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
epochs = range(1, len(history.history["loss"]) + 1)
for ax, key, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")]:
    ax.plot(epochs, history.history[key], color="#2a78d6", lw=2, marker="o", ms=4, label="Training")
    ax.plot(epochs, history.history["val_" + key], color="#eb6834", lw=2, marker="o", ms=4, label="Validation")
    ax.set_title(title, color=INK, loc="left", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_xticks(list(epochs))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
plt.close(fig)

# Confusion matrix (row-normalised: "of all real X, what % did we predict as Y")
cm = confusion_matrix(y_test, lstm_pred, normalize="true")
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=30, ha="right")
ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
ax.set_xlabel("Predicted emotion")
ax.set_ylabel("Actual emotion")
ax.set_title("Confusion matrix - BiLSTM (test set)", color=INK, loc="left", fontweight="bold")
for i in range(len(EMOTIONS)):
    for j in range(len(EMOTIONS)):
        if cm[i, j] >= 0.01:
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else INK, fontsize=9)
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close(fig)

# Model comparison
fig, ax = plt.subplots(figsize=(7.2, 3.8))
names = ["TF-IDF + LogReg\n(machine learning)", "BiLSTM\n(deep learning)"]
x = np.arange(2)
for k, (label, vals, color) in enumerate([
    ("Test accuracy", [base_acc, lstm_acc], "#2a78d6"),
    ("Test macro F1", [base_f1, lstm_f1], "#eb6834"),
    ("Real-world check", [base_real, lstm_real], "#1baf7a"),
]):
    bars = ax.bar(x + (k - 1) * 0.26, vals, width=0.24, color=color, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", color=INK, fontsize=9)
ax.set_xticks(x, names)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("Baseline vs. deep learning", color=INK, loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
plt.close(fig)

print(f"\nSaved model to {MODEL_DIR} and charts to {RESULTS_DIR}. Now run:  streamlit run app.py")
