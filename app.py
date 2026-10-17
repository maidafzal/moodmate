"""
MoodMate - Step 2: the web app.

Write a journal entry -> the trained Bidirectional LSTM detects the emotion,
explains which words drove its decision, suggests something helpful, and
tracks your mood over time.

Run:  streamlit run app.py      (train the model first with: python train.py)
"""

import html
import json
import os
from datetime import datetime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "moodmate_lstm.keras")
VOCAB_PATH = os.path.join(HERE, "model", "vocab.json")
RESULTS_DIR = os.path.join(HERE, "results")
JOURNAL_PATH = os.path.join(HERE, "journal.csv")

# Everything the app says about each emotion. Colours are a colour-blind-checked palette.
EMOTION_INFO = {
    "sadness": {
        "emoji": "😢", "color": "#2a78d6", "headline": "Be gentle with yourself today.",
        "tips": ["Message one person you trust - even a short \"hey\" helps.",
                 "Take a 10-minute walk outside, no phone.",
                 "Write down three small things that went okay today."],
        "prompt": "What would you say to a friend who felt exactly like this?",
    },
    "joy": {
        "emoji": "😄", "color": "#eda100", "headline": "Savour this moment!",
        "tips": ["Write down exactly what made today good, so you can repeat it.",
                 "Share the good news with someone - joy grows when shared.",
                 "Take a photo or keep a small reminder of today."],
        "prompt": "Who or what helped make today a good day?",
    },
    "love": {
        "emoji": "🥰", "color": "#e87ba4", "headline": "Your heart is full - express it.",
        "tips": ["Tell the person how much they mean to you.",
                 "Write a short thank-you note (sending it is optional).",
                 "Plan a small thing to do together this week."],
        "prompt": "What is one thing you appreciate about this person?",
    },
    "anger": {
        "emoji": "😠", "color": "#e34948", "headline": "Let's cool down before reacting.",
        "tips": ["Try box breathing: in 4 s, hold 4 s, out 4 s, hold 4 s - repeat 4 times.",
                 "Step away from the situation for 15 minutes.",
                 "Write an 'unsent letter' saying everything you feel - then delete it."],
        "prompt": "What need of yours was not respected here?",
    },
    "fear": {
        "emoji": "😰", "color": "#4a3aa7", "headline": "You are safe right now. Let's ground you.",
        "tips": ["5-4-3-2-1: name 5 things you see, 4 you hear, 3 you can touch, 2 you smell, 1 you taste.",
                 "Break the scary task into one tiny first step and do only that.",
                 "Remind yourself of a time you handled something hard."],
        "prompt": "What is the worst, best and most likely thing that could happen?",
    },
    "surprise": {
        "emoji": "😲", "color": "#1baf7a", "headline": "Something unexpected happened!",
        "tips": ["Pause and take three slow breaths before deciding anything.",
                 "Write down what you expected versus what actually happened.",
                 "Ask yourself: is this a good surprise, a bad one, or not sure yet?"],
        "prompt": "How does this change your plans or how you see things?",
    },
}
EMOTIONS = list(EMOTION_INFO)

EXAMPLES = {
    "Pick an example...": "",
    "Proud day": "i finally finished my project and my teacher loved it, i feel so proud and happy today",
    "Missing someone": "i miss my grandmother so much, the house feels empty and quiet without her",
    "Exam nerves": "my exam is tomorrow and i feel so nervous and scared that i cant even sleep",
    "Annoying roommate": "i am so annoyed that my roommate ate my food again without even asking",
    "Family evening": "spending the evening with my family made me feel so loved and grateful",
    "Unexpected result": "i was shocked when i saw my result, i never expected to get the top grade",
}

DEMO_WEEK = [
    "i feel so nervous and anxious about my presentation on monday",
    "the presentation went really well and i feel relieved and happy",
    "i feel lonely and sad because all my friends were busy today",
    "i feel so angry that my brother broke my headphones and did not even say sorry",
    "i was amazed when my old friend called me out of nowhere",
    "had dinner with my parents and i feel so loved and cared for",
    "i finally understood the topic i was struggling with and i feel great",
]

# Phrases that should always show a support message, whatever the model says.
CRISIS_PHRASES = ["suicide", "kill myself", "end my life", "want to die", "self harm",
                  "self-harm", "hurt myself", "no reason to live", "better off without me"]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Waking up the AI model...")
def load_model():
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH)
    with open(VOCAB_PATH) as f:
        meta = json.load(f)
    vectorizer = keras.layers.TextVectorization(
        vocabulary=meta["vocab"], output_sequence_length=meta["max_len"])
    return tf, model, vectorizer, meta["max_len"]


def predict(texts):
    """List of sentences -> array of shape (n, 6) with one probability per emotion."""
    tf, model, vectorizer, _ = load_model()
    return model(vectorizer(tf.constant(texts)), training=False).numpy()


def explain(text, emotion_idx):
    """Explainable AI: remove each word in turn and measure how much the
    confidence drops. Big drop = that word mattered."""
    max_len = load_model()[3]
    words = text.split()[:max_len]
    if len(words) < 2:
        return [(w, 0.0) for w in words]
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = predict(variants)[:, emotion_idx]
    return list(zip(words, probs[0] - probs[1:]))


# ----------------------------------------------------------------------------
# Journal (saved locally in journal.csv - never sent anywhere)
# ----------------------------------------------------------------------------
def read_journal():
    if os.path.exists(JOURNAL_PATH):
        df = pd.read_csv(JOURNAL_PATH, parse_dates=["time"])
        if not df.empty:
            return df
    return pd.DataFrame(columns=["time", "entry", "emotion", "confidence"])


def add_entries(rows):
    df = pd.concat([read_journal(), pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(JOURNAL_PATH, index=False)


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit versions
        st.altair_chart(chart, use_container_width=True)


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def highlight_html(scored_words, color):
    top = max([s for _, s in scored_words] + [1e-6])
    spans = []
    for word, score in scored_words:
        alpha = 0.6 * max(score, 0) / top
        spans.append(f'<span class="mm-word" title="{score:+.0%} confidence" '
                     f'style="background:{hex_to_rgba(color, alpha)}">{html.escape(word)}</span>')
    return "".join(spans)


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=DM+Sans:wght@400;500;700&display=swap');

.stApp {
  background:
    radial-gradient(900px 500px at 0% -10%, #ece4ff 0%, rgba(236,228,255,0) 60%),
    radial-gradient(800px 480px at 100% 0%, #ffe8dc 0%, rgba(255,232,220,0) 55%),
    #fbf8f4;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #f3eefc, #fbf3ee); border-right: 1px solid #eee7f5; }
.stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input, .stApp button,
.stApp [data-testid="stMarkdownContainer"] { font-family: 'DM Sans', sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { font-family: 'Fraunces', serif; font-weight: 600; color: #2b2540; }
.block-container { padding-top: 2.5rem; max-width: 760px; }

/* header */
.mm-hero { display: flex; align-items: center; gap: 18px; padding: 4px 0 20px; }
.mm-moon { flex: none; font-size: 40px; width: 72px; height: 72px; border-radius: 22px; display: grid; place-items: center;
  background: linear-gradient(145deg, #2f2a55, #6b5bd2); box-shadow: 0 12px 30px rgba(74,58,167,.28); }
.mm-greet { color: #8a7fb0; font-size: .92rem; letter-spacing: .02em; }
.mm-title { font-family: 'Fraunces', serif; font-size: 2.6rem; font-weight: 600; line-height: 1.05; color: #2b2540; }
.mm-sub { color: #6e6787; margin-top: 2px; }

/* tabs as soft pills */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(255,255,255,.7); padding: 5px; border-radius: 999px; border: 1px solid #ece6f5; }
.stTabs [data-baseweb="tab"] { height: auto; padding: 7px 16px; border-radius: 999px; background: transparent; }
.stTabs [aria-selected="true"] { background: #fff; box-shadow: 0 2px 10px rgba(43,37,64,.08); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.3rem; }

/* inputs and buttons */
.stTextArea [data-baseweb="textarea"], .stSelectbox [data-baseweb="select"] > div { border-radius: 16px; background: #fff; border-color: #e9e3f3; }
.stTextArea textarea { font-size: 1.05rem; line-height: 1.6; padding: 14px 16px; background: #fff; }
.stButton > button { border-radius: 999px; padding: .5rem 1.3rem; border: 1px solid #e3dcf0; background: #fff; font-weight: 500; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #6b5bd2, #a184ee); border: none; color: #fff; box-shadow: 0 8px 22px rgba(107,91,210,.30); }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { filter: brightness(1.06); color: #fff; }

/* cards */
.mm-card { background: rgba(255,255,255,.85); border: 1px solid #efe9f6; border-radius: 20px; padding: 20px 22px;
  box-shadow: 0 6px 24px rgba(43,37,64,.06); margin: 4px 0 18px; }
.mm-result { display: flex; gap: 18px; align-items: center; border-left: 6px solid var(--accent); }
.mm-emoji { font-size: 50px; line-height: 1; }
.mm-kicker { color: #8a7fb0; font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; }
.mm-emotion { font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 600; color: #2b2540; text-transform: capitalize; line-height: 1.15; }
.mm-pill { display: inline-block; margin-top: 6px; padding: 3px 12px; border-radius: 999px; font-size: .85rem; font-weight: 500; color: #2b2540; }
.mm-h { font-family: 'Fraunces', serif; font-size: 1.2rem; font-weight: 600; color: #2b2540; margin-bottom: 2px; }
.mm-cap { color: #8a7fb0; font-size: .88rem; margin-bottom: 12px; }
.mm-word { display: inline-block; padding: 3px 10px; margin: 3px 2px; border-radius: 999px; font-size: 1.02rem; }
.mm-tips { list-style: none; padding: 0; margin: 10px 0 14px; }
.mm-tips li { padding: 9px 0 9px 28px; position: relative; border-bottom: 1px dashed #eee7f5; margin: 0; }
.mm-tips li:last-child { border-bottom: none; }
.mm-tips li::before { content: "✦"; position: absolute; left: 4px; top: 9px; color: var(--accent); }
.mm-reflect { border-radius: 14px; padding: 12px 16px; font-style: italic; color: #3b3456; }

[data-testid="stMetric"] { background: rgba(255,255,255,.85); border: 1px solid #efe9f6; border-radius: 18px; padding: 14px 16px; }
[data-testid="stAlert"] { border-radius: 16px; }
[data-testid="stExpander"] details { border-radius: 16px; background: rgba(255,255,255,.7); }
[data-testid="stImage"] img, .stImage img { border-radius: 16px; }
</style>
"""


def emotion_color_scale():
    return alt.Scale(domain=EMOTIONS, range=[EMOTION_INFO[e]["color"] for e in EMOTIONS])


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MoodMate - AI feelings journal", page_icon="🌙", layout="centered")

if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python train.py` first, then restart the app.")
    st.stop()

st.markdown(STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="mm-title" style="font-size:1.7rem">🌙 MoodMate</div>', unsafe_allow_html=True)
    st.write("An AI journal that reads the emotion in your writing, explains why, "
             "and helps you notice your mood patterns over time.")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        st.metric("Model accuracy on unseen text", f"{m['bilstm']['accuracy']:.1%}")
        st.caption("Model: Bidirectional LSTM (Recurrent Neural Network) trained on "
                   f"{m['train_size']:,} labelled sentences.")

hour = datetime.now().hour
greeting = "Good morning ☀️" if hour < 12 else "Good afternoon 🌤️" if hour < 17 else "Good evening ✨"
st.markdown(f"""
<div class="mm-hero">
  <div class="mm-moon">🌙</div>
  <div>
    <div class="mm-greet">{greeting}</div>
    <div class="mm-title">MoodMate</div>
    <div class="mm-sub">a gentle AI journal that listens to how you feel</div>
  </div>
</div>""", unsafe_allow_html=True)

tab_write, tab_journey, tab_how = st.tabs(["✍️ Write", "📈 My mood journey", "🧠 How the AI works"])

# --- Tab 1: write ------------------------------------------------------------
with tab_write:
    def use_example():
        st.session_state.entry = EXAMPLES[st.session_state.example]

    st.selectbox("Need a spark? Try an example", list(EXAMPLES),
                 key="example", on_change=use_example)
    text = st.text_area("How was your day?", key="entry", height=140,
                        placeholder="Dear diary, today i felt...")

    if st.button("✨ Analyse my mood", type="primary"):
        if len(text.split()) < 3:
            st.warning("Write at least a short sentence (3+ words) so the AI has something to read.")
            st.session_state.pop("result", None)
        else:
            probs = predict([text])[0]
            idx = int(probs.argmax())
            st.session_state.result = {
                "text": text, "probs": probs.tolist(), "idx": idx,
                "words": [(w, float(s)) for w, s in explain(text, idx)], "saved": False,
            }

    res = st.session_state.get("result")
    if res and res["text"] == text:
        lowered = text.lower()
        if any(p in lowered for p in CRISIS_PHRASES):
            st.error("💛 **It sounds like you're going through something really painful.** "
                     "You don't have to handle it alone. Please talk to someone you trust right now, "
                     "or contact a local crisis helpline or emergency services.")

        probs = np.array(res["probs"])
        emotion = EMOTIONS[res["idx"]]
        info = EMOTION_INFO[emotion]
        conf = probs[res["idx"]]

        tint = hex_to_rgba(info["color"], 0.14)
        if conf < 0.5:
            second = EMOTIONS[int(np.argsort(probs)[-2])]
            note = f"only {conf:.0%} sure &middot; feels mixed, with some {second} too"
        else:
            note = f"{conf:.0%} sure"
        st.markdown(f"""
<div class="mm-card mm-result" style="--accent:{info['color']}">
  <div class="mm-emoji">{info['emoji']}</div>
  <div>
    <div class="mm-kicker">you seem to be feeling</div>
    <div class="mm-emotion">{emotion}</div>
    <span class="mm-pill" style="background:{tint}">{note}</span>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="mm-h">Emotion breakdown</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({"emotion": EMOTIONS, "probability": probs})
        bars = alt.Chart(prob_df).mark_bar(cornerRadiusEnd=4, height=18).encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", title=None, grid=False)),
            y=alt.Y("emotion:N", sort=EMOTIONS, title=None),
            color=alt.Color("emotion:N", scale=emotion_color_scale(), legend=None),
            tooltip=[alt.Tooltip("emotion:N"), alt.Tooltip("probability:Q", format=".1%")],
        )
        labels = bars.mark_text(align="left", dx=4).encode(
            text=alt.Text("probability:Q", format=".0%"), color=alt.value("#52514e"))
        show_chart((bars + labels).properties(height=210))

        tips = "".join(f"<li>{tip}</li>" for tip in info["tips"])
        st.markdown(f"""
<div class="mm-card">
  <div class="mm-h">🔍 Why MoodMate thinks so</div>
  <div class="mm-cap">Darker highlight = the word pushed the AI more towards <i>{emotion}</i>. Hover a word to see its effect.</div>
  <div>{highlight_html(res["words"], info["color"])}</div>
</div>
<div class="mm-card" style="--accent:{info['color']}">
  <div class="mm-h">💡 {info['headline']}</div>
  <ul class="mm-tips">{tips}</ul>
  <div class="mm-reflect" style="background:{tint}">Reflect: {info['prompt']}</div>
</div>""", unsafe_allow_html=True)

        if res["saved"]:
            st.success("Saved to your journal. See it in **My mood journey**.")
        elif st.button("💾 Save to my journal"):
            add_entries([{"time": datetime.now().replace(microsecond=0), "entry": text,
                          "emotion": emotion, "confidence": round(float(conf), 3)}])
            res["saved"] = True
            st.rerun()

# --- Tab 2: mood journey -----------------------------------------------------
with tab_journey:
    journal = read_journal()

    if journal.empty:
        st.info("Your journal is empty. Save an entry from the **Write** tab, "
                "or load a demo week to see how mood tracking works.")
        if st.button("Load a demo week"):
            probs = predict(DEMO_WEEK)
            start = datetime.now().replace(hour=21, minute=0, second=0, microsecond=0) - timedelta(days=6)
            add_entries([{"time": start + timedelta(days=i), "entry": t,
                          "emotion": EMOTIONS[int(p.argmax())], "confidence": round(float(p.max()), 3)}
                         for i, (t, p) in enumerate(zip(DEMO_WEEK, probs))])
            st.rerun()
    else:
        counts = journal["emotion"].value_counts()
        top = counts.index[0]
        positive = journal["emotion"].isin(["joy", "love"]).mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Entries", len(journal))
        c2.metric("Most frequent mood", f"{EMOTION_INFO[top]['emoji']} {top}")
        c3.metric("Positive entries", f"{positive:.0%}")

        st.markdown("##### Your moods over time")
        timeline = alt.Chart(journal).mark_circle(size=220, opacity=0.9, stroke="white", strokeWidth=2).encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b", grid=False)),
            y=alt.Y("emotion:N", sort=EMOTIONS, title=None, scale=alt.Scale(domain=EMOTIONS)),
            color=alt.Color("emotion:N", scale=emotion_color_scale(), legend=None),
            tooltip=[alt.Tooltip("time:T", format="%a %d %b, %H:%M"), "emotion:N",
                     alt.Tooltip("confidence:Q", format=".0%"), "entry:N"],
        ).properties(height=260)
        show_chart(timeline)

        st.markdown("##### How often you felt each emotion")
        count_df = counts.reindex(EMOTIONS, fill_value=0).rename_axis("emotion").reset_index(name="entries")
        count_chart = alt.Chart(count_df).mark_bar(cornerRadiusEnd=4, height=18).encode(
            x=alt.X("entries:Q", title=None, axis=alt.Axis(tickMinStep=1, grid=False)),
            y=alt.Y("emotion:N", sort=EMOTIONS, title=None),
            color=alt.Color("emotion:N", scale=emotion_color_scale(), legend=None),
            tooltip=["emotion:N", "entries:Q"],
        ).properties(height=210)
        show_chart(count_chart)

        hard = journal[journal["emotion"].isin(["sadness", "fear", "anger"])]
        if len(hard) / len(journal) > 0.6:
            st.warning("Most of your recent entries show difficult emotions. That's okay - "
                       "but consider talking to a friend, family member or counsellor about how you feel.")
        elif top in ("joy", "love"):
            st.success(f"You've mostly felt **{top}** lately. Keep writing - patterns become clearer over time.")
        else:
            st.info(f"You've mostly felt **{top}** lately. Look back at those entries - "
                    "what was happening on those days?")

        with st.expander("See all entries"):
            st.dataframe(journal.sort_values("time", ascending=False), hide_index=True)
        if st.button("🗑️ Clear my journal"):
            os.remove(JOURNAL_PATH)
            st.rerun()

# --- Tab 3: how it works -----------------------------------------------------
with tab_how:
    st.markdown("""
#### From your words to an emotion

1. **Clean the text** - lowercase it and strip punctuation.
2. **Tokenise** - turn each word into a number using a 20,000-word vocabulary.
3. **Embedding layer** - turn each number into a 128-value vector that captures meaning
   (words like *happy* and *glad* end up close together).
4. **Bidirectional LSTM** - a Recurrent Neural Network reads the sentence forwards *and*
   backwards, so every word is understood together with the words around it.
5. **Softmax layer** - outputs one probability for each of the 6 emotions.
6. **Explainability** - each word is removed in turn; the bigger the drop in confidence,
   the more that word mattered.
""")
    if os.path.exists(metrics_path):
        st.markdown("#### Results")
        base, lstm = m["baseline_tfidf_logreg"], m["bilstm"]
        st.table(pd.DataFrame({
            "Model": ["TF-IDF + Logistic Regression (machine learning)", "Bidirectional LSTM (deep learning)"],
            "Test accuracy": [f"{base['accuracy']:.1%}", f"{lstm['accuracy']:.1%}"],
            "Macro F1": [f"{base['macro_f1']:.1%}", f"{lstm['macro_f1']:.1%}"],
            f"Real-world check ({m['real_world_size']} own sentences)": [
                f"{base['real_world']:.0%}", f"{lstm['real_world']:.0%}"],
        }).set_index("Model"))
        st.caption(f"Test set = {m['test_size']:,} sentences the model never saw during training. "
                   "The real-world check uses sentences we wrote ourselves - it is lower, which shows "
                   "how benchmark scores can be optimistic.")
    for name, caption in [("model_comparison.png", "Accuracy and macro-F1 on 2,000 unseen test sentences"),
                          ("confusion_matrix.png", "Where the model gets confused (rows = true emotion)"),
                          ("training_curves.png", "Learning progress during training")]:
        path = os.path.join(RESULTS_DIR, name)
        if os.path.exists(path):
            st.image(path, caption=caption)
    st.markdown("""
#### Responsible AI
- **Not a diagnosis.** The model detects emotion in text; it does not understand your life.
- **Safety first.** Crisis phrases always trigger a support message, whatever the model predicts.
- **Privacy.** Everything runs on your computer - no text is sent to any server.
- **Known limits.** Trained on short English sentences that mostly start with *"i feel..."*, so it is weaker on everyday phrasing, negation (*"not happy"*), sarcasm, slang and other languages.
""")
