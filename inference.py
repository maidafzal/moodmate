"""
MoodMate inference engine - runs the trained Bidirectional LSTM with plain NumPy.

TensorFlow is only needed for training (train.py). For the web app we load the
learned weights and compute the same LSTM equations by hand. This keeps the app
small, fast to start and installable on any Python version.
"""

import json
import re

import numpy as np

# Same punctuation Keras TextVectorization removes by default
STRIP_PUNCTUATION = re.compile(r'[!"#$%&()\*\+,\-\./:;<=>?@\[\\\]^_`{|}~\']')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class EmotionModel:
    def __init__(self, weights_path, vocab_path):
        with open(vocab_path) as f:
            meta = json.load(f)
        self.emotions = meta["emotions"]
        self.max_len = meta["max_len"]
        self.word_index = {w: i for i, w in enumerate(meta["vocab"])}
        self.w = dict(np.load(weights_path))

    def tokenize(self, text):
        """Lowercase, strip punctuation, split on spaces, look up each word.
        Unknown words get ID 1 ("[UNK]"), exactly like during training."""
        words = STRIP_PUNCTUATION.sub("", text.lower()).split()
        return [self.word_index.get(w, 1) for w in words[: self.max_len]]

    def _lstm(self, direction, sequences):
        """Run one LSTM direction over a batch of word-ID lists and return the
        hidden state after the last real word of each sentence."""
        kernel = self.w[f"{direction}_kernel"]               # input  -> 4 gates
        recurrent = self.w[f"{direction}_recurrent_kernel"]  # memory -> 4 gates
        bias = self.w[f"{direction}_bias"]
        units = recurrent.shape[0]

        steps = max([len(s) for s in sequences] + [1])
        ids = np.zeros((len(sequences), steps), dtype=np.int64)
        for row, seq in enumerate(sequences):
            ids[row, :len(seq)] = seq
        mask = ids != 0                                      # padding is skipped
        x = self.w["embedding"][ids]                         # word IDs -> meaning vectors

        h = np.zeros((len(sequences), units))                # hidden state (short-term memory)
        c = np.zeros((len(sequences), units))                # cell state (long-term memory)
        for t in range(steps):
            z = x[:, t] @ kernel + h @ recurrent + bias
            i, f, g, o = np.split(z, 4, axis=1)              # input, forget, candidate, output
            c_new = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
            h_new = sigmoid(o) * np.tanh(c_new)
            keep = mask[:, t:t + 1]
            c = np.where(keep, c_new, c)
            h = np.where(keep, h_new, h)
        return h

    def predict(self, texts):
        """List of sentences -> array of shape (n, 6) with one probability per emotion."""
        seqs = [self.tokenize(t) for t in texts]
        forward = self._lstm("forward", seqs)
        backward = self._lstm("backward", [s[::-1] for s in seqs])  # read right -> left
        hidden = np.concatenate([forward, backward], axis=1)
        hidden = np.maximum(0, hidden @ self.w["dense_kernel"] + self.w["dense_bias"])  # ReLU
        logits = hidden @ self.w["output_kernel"] + self.w["output_bias"]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))                        # softmax
        return exp / exp.sum(axis=1, keepdims=True)
