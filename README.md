# 🌙 MoodMate — an AI feelings journal

MoodMate is a journal app that **reads the emotion in what you write**, shows **which words made it decide**, suggests **something helpful to do**, and tracks **your mood over the week**.

Under the hood it uses a **Bidirectional LSTM**, a type of Recurrent Neural Network (RNN). It was trained on about 390,000 labelled sentences and is compared against a classical machine-learning baseline.

| Write an entry | AI explains itself | Mood over time |
|---|---|---|
| Type how your day went (or pick an example) | Words that pushed the decision are highlighted | Timeline + counts of your emotions |

---

## 1. Problem & motivation

Many students feel stressed, lonely or anxious, but they rarely stop to name the feeling or notice patterns ("I'm always anxious on Sundays"). Journaling helps, but a plain diary gives no feedback.

**Goal:** build an AI that turns a short diary entry into:
1. the emotion it expresses (one of 6: sadness, joy, love, anger, fear, surprise),
2. an explanation of *why* the AI thinks so,
3. a small, practical suggestion for that emotion,
4. a history of moods over time.

## 2. AI concepts used

| Concept | Where |
|---|---|
| Natural Language Processing (NLP) | Cleaning text, tokenising words into numbers |
| Word embeddings | Each word becomes a 128-number "meaning vector" learned during training |
| Recurrent Neural Network: **Bidirectional LSTM** | Reads the sentence forwards and backwards to understand context |
| Classical ML baseline: TF-IDF + Logistic Regression | Shows whether deep learning is really better |
| Regularisation: Dropout, Spatial Dropout, Label Smoothing, Early Stopping | Prevents over-fitting and over-confidence |
| Explainable AI: occlusion | Remove one word at a time and see how much the confidence drops |
| LSTM equations by hand (NumPy) | The web app runs the trained network with plain NumPy (`inference.py`), so it needs no TensorFlow; it matches TensorFlow's output on all 2,000 test sentences |
| Responsible AI | Crisis-phrase safety message, privacy (runs locally), honest limitations |

## 3. Dataset

**Emotion dataset** — Saravia et al., 2018, *CARER: Contextualized Affect Representations for Emotion Recognition* (EMNLP). It is downloaded automatically from Hugging Face (`dair-ai/emotion`). The dataset card says it is for educational and research use.

- 416,809 English sentences, each labelled with one emotion.
- The standard **2,000-sentence validation set** and **2,000-sentence test set** were **removed from the training data**, along with duplicates. This leaves **389,827 training sentences**, so the test score is truly on unseen text.
- The classes are imbalanced: joy 34%, sadness 30%, anger 14%, fear 11%, love 7.5%, surprise 3%.

## 4. Model

```
"i miss my grandmother"                       raw text
        │  TextVectorization (lowercase, strip punctuation, 20,000-word vocabulary)
        ▼
[ 2, 104, 9, 3311, 0, 0, ... ]                50 word IDs (padded)
        │  Embedding(20,000 → 128)            each word → 128-number vector
        │  SpatialDropout1D(0.2)
        │  Bidirectional LSTM(64 + 64)        reads left→right and right→left
        │  Dropout(0.5) → Dense(64, ReLU)
        ▼  Dense(6, Softmax)
[ sadness 0.89, joy 0.02, love 0.02, ... ]    one probability per emotion
```

- 2.67 million trainable parameters.
- Optimiser: Adam. Loss: categorical cross-entropy with label smoothing 0.1. Batch size: 256.
- Early stopping on validation accuracy. Training stopped after 4 epochs, about 3 minutes on a laptop CPU.

## 5. Results

| Model | Test accuracy | Macro F1 | Real-world check (24 own sentences) |
|---|---|---|---|
| TF-IDF + Logistic Regression (ML baseline) | 90.8% | 85.9% | 67% |
| **Bidirectional LSTM (MoodMate)** | **94.0%** | **90.1%** | 67% |

Per-emotion results for the LSTM (test set):

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| sadness | 0.99 | 0.96 | 0.98 |
| joy | 0.95 | 0.96 | 0.96 |
| love | 0.85 | 0.81 | 0.83 |
| anger | 0.92 | 0.96 | 0.94 |
| fear | 0.88 | 0.96 | 0.92 |
| surprise | 0.87 | 0.71 | 0.78 |

Charts are in `results/`: `model_comparison.png`, `confusion_matrix.png` and `training_curves.png`.

**What the results tell us**
- Deep learning beats the baseline by **+3.2 points of accuracy** and **+4.2 points of macro F1**.
- The hardest emotions are the rarest ones. **Love is confused with joy** (19% of love sentences), and **surprise is confused with fear** (26%). Both pairs genuinely overlap in real language.
- The **real-world check** uses 24 sentences we wrote ourselves, and it is much lower (67%). The dataset sentences mostly look like *"i feel …"*, so the model struggles with other phrasing and with negation (*"i am not happy at all"* → joy). This gap between benchmark and real life is an important finding, not a failure. See `results/classification_report.txt` for every sentence.

## 6. App features

1. **Emotion detection** with a probability bar for all 6 emotions. If the model is less than 50% sure, it says the entry feels mixed.
2. **"Why MoodMate thinks so"**: the words that pushed the decision are highlighted. This is explainable AI.
3. **Suggestion card**: three practical ideas and a reflection question for each emotion (e.g. the 5-4-3-2-1 grounding technique for fear, box breathing for anger).
4. **Mood journey**: saved entries over time, how often you felt each emotion, and the share of positive entries. A **"Load a demo week"** button fills it for presentations.
5. **How the AI works** tab: the pipeline, the results table and the charts, ready to show your teacher.
6. **Safety**: phrases such as "want to die" or "hurt myself" always show a support message, whatever the model predicts.
7. **Privacy**: entries are never saved on a server. Each visitor's journal lives only in their browser session, and they can download it as a CSV or clear it.

## 7. How to run

The app runs on any recent Python (3.9 or newer). Retraining needs TensorFlow, which supports **Python 3.10–3.12**. Check with `python --version`.

**Windows**
```bat
cd MoodMate-AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Mac / Linux**
```bash
cd MoodMate-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. A trained model is already included in `model/`.

**To retrain the model yourself** (about 5–10 min; downloads about 30 MB of data the first time):
```bash
pip install -r requirements-train.txt
python train.py
```

**Google Colab**: upload `train.py`, then run `!pip install -q pyarrow` followed by `!python train.py`. Download the `model/` and `results/` folders it creates.

**Troubleshooting**
- On a Mac, if TensorFlow crashes with `mutex lock failed`, run `pip install "tensorflow==2.19.*"`.
- If you see `No trained model found`, run `python train.py` first.

**Deploy online (free) with Streamlit Community Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository `maidafzal/moodmate`, branch `main`, main file `app.py`.
4. No secrets or environment variables are needed, and any Python version works.
5. Click **Deploy**. The first build takes a few minutes while TensorFlow installs.

## 8. Project structure

```
MoodMate-AI/
├── train.py          # downloads data, trains baseline + BiLSTM, saves model and charts
├── app.py            # Streamlit web app
├── inference.py      # runs the trained LSTM with plain NumPy (used by the app)
├── requirements.txt        # app only (no TensorFlow)
├── requirements-train.txt  # extra libraries for train.py
├── model/            # moodmate_lstm.keras (Keras model), weights.npz (for the app), vocab.json
├── results/          # metrics.json, classification_report.txt, charts
└── .streamlit/       # app theme
```

## 9. Limitations & future work

**Limitations**
- English only. The model is weaker on everyday phrasing, sarcasm, slang and negation.
- Each entry gets one emotion, but real feelings are often mixed.
- It is **not a medical or diagnostic tool**.

**Future work**
- Fine-tune a pretrained Transformer (e.g. DistilBERT) to handle negation and natural phrasing better.
- Multi-label emotions, so one entry can be "joy + fear".
- Urdu / Roman-Urdu support.
- Voice journaling with speech-to-text.
- Weekly email summaries of mood trends.

## 10. Viva preparation — likely questions

**Why an LSTM and not a simple neural network?**
Word order matters: "not happy" ≠ "happy not". An LSTM reads words in sequence and keeps a memory of earlier words. A plain dense network on a bag of words ignores order.

**Why *bidirectional*?**
A word's meaning can depend on words that come after it. Reading in both directions gives each word context from both sides.

**What is an embedding?**
A learned vector for each word. Words used in similar contexts (happy / glad) get similar vectors, so the model generalises between them.

**What is over-fitting and how did you handle it?**
Over-fitting means memorising the training data instead of learning general patterns. We used Dropout, Spatial Dropout, Label Smoothing, Early Stopping, and a validation set kept separate from training.

**Why report macro F1 and not just accuracy?**
The classes are imbalanced (surprise is only 3%). A model could score high accuracy while ignoring rare emotions. Macro F1 averages every emotion equally.

**Why is the real-world score lower than the test score?**
The test set comes from the same source as the training data. Our own sentences are written differently. This is called *distribution shift*, and it is why AI must be tested on realistic data.

**How does the explanation (highlighting) work?**
Occlusion. We remove one word at a time and re-run the model. The bigger the drop in confidence, the more that word mattered.

**What is label smoothing?**
Instead of training towards 100% certainty, we train towards 90%. This makes the confidence scores shown in the app more honest.

**How is user privacy protected?**
Entries are analysed in memory and never stored on the server. Each visitor's journal exists only in their own browser session, and they can download it or clear it with one button.

## References
- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). *CARER: Contextualized Affect Representations for Emotion Recognition.* EMNLP.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
- TensorFlow / Keras, scikit-learn and Streamlit documentation.
