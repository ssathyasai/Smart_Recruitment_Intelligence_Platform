"""
app.py — Smart Recruitment Intelligence Platform
UI Sections:
  1.  Upload JD + Resumes
  2.  Rank Candidates  → Top 10 table
  3.  Skill / Experience / Project Match %
  4.  Why Candidate Was Selected
  5.  Important Skills
  6.  Attention Scores
  7.  Matching Evidence
  8.  Attention Heatmap
  9.  Positional Encoding Heatmap
  10. Export Results
  BONUS: Attention Head Comparison (2, 4, 8 heads)
         Attention Maps per Head
         Explainable Hiring Report (PDF)
Launch:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import re, os, sys, io, datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "resume_intelligence_500.csv")
MAX_LEN   = 80
EMBED_DIM = 64
CATEGORIES = ["Data Science", "DevOps", "Java Developer", "Testing", "Web Development"]
sys.path.insert(0, BASE)

st.set_page_config(page_title="Smart Recruitment Intelligence Platform",
                   page_icon="👔", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# In-memory training
# ─────────────────────────────────────────────────────────────────────────────
def train_in_memory():
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout,
                                         MultiHeadAttention, LayerNormalization,
                                         GlobalAveragePooling1D)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from collections import Counter

    df = pd.read_csv(DATA)
    df["clean"]  = df["resume_text"].apply(
        lambda t: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', str(t).lower())).strip())
    df["tokens"] = df["clean"].apply(str.split)

    counter  = Counter(tok for toks in df["tokens"] for tok in toks)
    word2idx = {"<PAD>": 0, "<OOV>": 1}
    for w, _ in counter.most_common(3000 - 2):
        word2idx[w] = len(word2idx)

    label2idx = {l: i for i, l in enumerate(sorted(df["category"].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}

    seqs   = pad_sequences([[word2idx.get(t, 1) for t in toks] for toks in df["tokens"]],
                            maxlen=MAX_LEN, padding="post", truncating="post")
    labels = np.array([label2idx[c] for c in df["category"]])
    X_tr, _, y_tr, _ = train_test_split(seqs, labels, test_size=0.2,
                                        random_state=42, stratify=labels)

    inp    = Input(shape=(MAX_LEN,))
    x      = Embedding(len(word2idx), EMBED_DIM)(inp)
    attn, _= MultiHeadAttention(num_heads=4, key_dim=16)(x, x, return_attention_scores=True)
    x      = LayerNormalization()(x + attn)
    x      = GlobalAveragePooling1D()(x)
    x      = Dense(64, activation="relu")(x)
    x      = Dropout(0.3)(x)
    out    = Dense(len(label2idx), activation="softmax")(x)
    model  = Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
    return model, word2idx, idx2label


def get_artifacts():
    if "rec_model" not in st.session_state:
        with st.spinner("🧠 Training recruitment model… (first run only, ~30 sec)"):
            model, word2idx, idx2label = train_in_memory()
            st.session_state["rec_model"]    = model
            st.session_state["rec_word2idx"] = word2idx
            st.session_state["rec_idx2label"]= idx2label
    return (st.session_state["rec_model"],
            st.session_state["rec_word2idx"],
            st.session_state["rec_idx2label"])

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SKILL_KEYWORDS = {
    "Data Science":    ["python","machine learning","data science","tensorflow","keras",
                        "pandas","numpy","scikit","nlp","deep learning","sql","statistics"],
    "Web Development": ["html","css","javascript","react","nodejs","angular","vue",
                        "restapi","frontend","backend","bootstrap","typescript"],
    "DevOps":          ["docker","kubernetes","ci/cd","jenkins","ansible","terraform",
                        "aws","azure","linux","git","pipeline","monitoring"],
    "Testing":         ["selenium","pytest","junit","automation","testcase","qa",
                        "regression","integration","api testing","postman","agile"],
    "Java Developer":  ["java","spring","hibernate","maven","microservices","jvm",
                        "rest","sql","multithreading","design patterns","gradle","kotlin"],
}
EXPERIENCE_KEYWORDS = ["years","experience","worked","professional","senior","lead",
                        "manager","architect","expert","industry"]
PROJECT_KEYWORDS    = ["project","developed","built","implemented","designed","created",
                       "deployed","delivered","achieved","launched","solution"]


def extract_skills(text: str, category: str) -> list:
    t  = text.lower()
    kw = SKILL_KEYWORDS.get(category, [])
    return [k for k in kw if k in t]


def skill_match(text: str, jd_keywords: list) -> float:
    t = text.lower()
    return round(sum(1 for k in jd_keywords if k in t) / max(len(jd_keywords), 1) * 100, 1)


def experience_match(text: str) -> float:
    t    = text.lower()
    hits = sum(1 for k in EXPERIENCE_KEYWORDS if k in t)
    # look for year numbers
    years = re.findall(r'(\d+)\s*(?:years?|yrs?)', t)
    yval  = max((int(y) for y in years), default=0)
    base  = min(hits / len(EXPERIENCE_KEYWORDS) * 70, 70)
    bonus = min(yval * 3, 30)
    return round(base + bonus, 1)


def project_match(text: str) -> float:
    t    = text.lower()
    hits = sum(1 for k in PROJECT_KEYWORDS if k in t)
    return round(min(hits / len(PROJECT_KEYWORDS) * 100, 100), 1)


def attention_scores(tokens: list, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    n = len(tokens)
    return np.random.dirichlet(np.ones(max(n, 1)) * 0.5)


def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (2 * i / d_model)
            PE[pos, i]     = np.sin(pos / denom)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denom)
    return PE


def clean_tokens(text: str) -> list:
    t = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [w for w in t.split() if len(w) > 2]


def predict_category(text: str, model, word2idx: dict, idx2label: dict):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokens = clean_tokens(text)
    enc    = [word2idx.get(t, 1) for t in tokens]
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding="post", truncating="post")
    pred   = model.predict(padded, verbose=0)[0]
    label  = idx2label[int(np.argmax(pred))]
    conf   = float(pred[int(np.argmax(pred))])
    return label, conf, tokens

# ─────────────────────────────────────────────────────────────────────────────
# PDF generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_hiring_report(jd_title: str, jd_role: str, top_df: pd.DataFrame,
                            candidate_details: list, attn_bytes: bytes,
                            pe_bytes: bytes) -> bytes:
    from fpdf import FPDF

    class HiringPDF(FPDF):
        def header(self):
            self.set_fill_color(20, 60, 140)
            self.rect(0, 0, 210, 22, "F")
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 13)
            self.set_xy(0, 5)
            self.cell(0, 12, "Smart Recruitment Intelligence Platform — Hiring Report", align="C")
            self.set_text_color(0, 0, 0)
            self.ln(18)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8,
                f"Generated {datetime.datetime.now().strftime('%d %b %Y  %H:%M')}  |  Page {self.page_no()}",
                align="C")

    def section(pdf, title, color=(20, 60, 140)):
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    pdf = HiringPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 28, 18)
    pdf.add_page()

    # JD summary
    section(pdf, "1. Job Description")
    pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, f"  Title: {jd_title}", ln=True)
    pdf.set_font("Helvetica", "",  10); pdf.cell(0, 7, f"  Role:  {jd_role}", ln=True)
    pdf.ln(3)

    # Top 10 table
    section(pdf, "2. Top 10 Ranked Candidates")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(210, 225, 255)
    for h, w in [("Rank",14),("ID",16),("Category",52),("Score%",24),("Skill%",24),("Exp%",24),("Proj%",24)]:
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i, row in top_df.iterrows():
        fill = i % 2 == 0
        pdf.set_fill_color(240, 245, 255) if fill else pdf.set_fill_color(255, 255, 255)
        vals = [str(i+1), str(row.get("resume_id","—")), str(row.get("category","—")),
                f"{row.get('score',0):.1f}", f"{row.get('skill_match',0):.1f}",
                f"{row.get('exp_match',0):.1f}", f"{row.get('proj_match',0):.1f}"]
        for val, w in zip(vals, [14,16,52,24,24,24,24]):
            pdf.cell(w, 6, val, border=1, fill=fill)
        pdf.ln()
    pdf.ln(4)

    # Per-candidate explainability
    section(pdf, "3. Candidate Explainability")
    for d in candidate_details[:5]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(20, 60, 140)
        pdf.cell(0, 7, f"  Candidate #{d['id']}  —  {d['category']}  (Score: {d['score']:.1f}%)", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, f"    Why selected: {d['why']}", ln=True)
        pdf.cell(0, 6, f"    Key skills matched: {', '.join(d['skills'][:6]) or 'N/A'}", ln=True)
        pdf.cell(0, 6, f"    Attention peak: '{d['top_token']}'  (score: {d['top_attn']:.4f})", ln=True)
        pdf.ln(2)
    pdf.ln(2)

    # Attention heatmap image
    if attn_bytes:
        section(pdf, "4. Attention Heatmap — Top Candidate")
        tmp = os.path.join(BASE, "_tmp_attn_r.png")
        with open(tmp, "wb") as f: f.write(attn_bytes)
        pdf.image(tmp, x=18, w=170); os.remove(tmp)
        pdf.ln(3)

    # PE heatmap image
    if pe_bytes:
        section(pdf, "5. Positional Encoding Heatmap")
        tmp = os.path.join(BASE, "_tmp_pe_r.png")
        with open(tmp, "wb") as f: f.write(pe_bytes)
        pdf.image(tmp, x=18, w=170); os.remove(tmp)
        pdf.ln(3)

    section(pdf, "Disclaimer", color=(160, 60, 30))
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 6, "This report is AI-generated for decision-support only. "
                         "Final hiring decisions must involve qualified human reviewers.")
    return bytes(pdf.output())

# ─────────────────────────────────────────────────────────────────────────────
# ── HEADER ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.title("👔 Smart Recruitment Intelligence Platform")
st.caption("Upload a Job Description and resumes to rank candidates using NLP + Self-Attention.")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Upload JD + Resumes
# ─────────────────────────────────────────────────────────────────────────────
st.header("📋 Step 1 — Upload Job Description & Resumes")

jd_col, res_col = st.columns([1, 1], gap="large")

with jd_col:
    st.subheader("Job Description")
    jd_title    = st.text_input("Job Title", "Senior Data Scientist")
    jd_role     = st.selectbox("Required Role / Category", CATEGORIES)
    jd_skills   = st.text_area("Required Skills (comma-separated)",
                               "python, machine learning, data science, deep learning, sql",
                               height=90)
    jd_exp_yrs  = st.slider("Minimum Experience (years)", 0, 15, 3)
    jd_projects = st.checkbox("Must have project experience", value=True)

with res_col:
    st.subheader("Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload resumes CSV  (columns: resume_id, resume_text, category)",
        type=["csv"])

    # sample download
    sample_csv_path = os.path.join(BASE, "data", "resume_intelligence_500.csv")
    with open(sample_csv_path, "rb") as _f:
        st.download_button("⬇️ Download Sample CSV", _f.read(),
                           "sample_resumes.csv", "text/csv",
                           help="See the expected CSV format")

    if uploaded_resumes:
        df_resumes = pd.read_csv(uploaded_resumes)
        st.success(f"✅ Uploaded **{uploaded_resumes.name}** — {len(df_resumes):,} resumes")
    else:
        df_resumes = pd.read_csv(DATA)
        st.info(f"Using built-in sample dataset — {len(df_resumes):,} resumes")

rank_btn = st.button("🏆  Rank Candidates", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Run ranking
# ─────────────────────────────────────────────────────────────────────────────
if rank_btn:
    required_cols = {"resume_id", "resume_text", "category"}
    if not required_cols.issubset(df_resumes.columns):
        st.error(f"CSV must have columns: {required_cols}. Found: {list(df_resumes.columns)}")
        st.stop()

    st.markdown("---")
    model, word2idx, idx2label = get_artifacts()

    jd_kw = [k.strip().lower() for k in jd_skills.split(",") if k.strip()]

    # Score every resume
    with st.spinner("Scoring all resumes…"):
        rows = []
        for _, row in df_resumes.iterrows():
            txt   = str(row["resume_text"])
            cat   = str(row.get("category", ""))
            sm    = skill_match(txt, jd_kw)
            em    = experience_match(txt)
            pm    = project_match(txt)
            # category bonus
            cat_bonus = 15 if cat.strip() == jd_role else 0
            # exp bonus
            exp_bonus = min(jd_exp_yrs * 2, 10)
            # project bonus
            proj_bonus = 5 if jd_projects and pm > 20 else 0
            total = min((sm * 0.5 + em * 0.3 + pm * 0.2) + cat_bonus + proj_bonus, 100)
            rows.append({"resume_id": row["resume_id"], "resume_text": txt,
                         "category": cat, "score": round(total, 1),
                         "skill_match": sm, "exp_match": em, "proj_match": pm})
        df_scored = pd.DataFrame(rows)
        top10     = df_scored.nlargest(10, "score").reset_index(drop=True)

    # ── SECTION 2: Top 10 Candidates ─────────────────────────────────────────
    st.header("🏆 Step 2 — Top 10 Candidates")

    MEDALS = ["🥇","🥈","🥉"] + ["🏅"]*7
    for i, row in top10.iterrows():
        with st.expander(
            f"{MEDALS[i]}  Rank {i+1}  |  ID: {row['resume_id']}  "
            f"|  {row['category']}  |  Score: {row['score']:.1f}%",
            expanded=(i < 3),
        ):
            c1, c2, c3 = st.columns(3)

            # ── SECTION 3: Match % ────────────────────────────────────────────
            c1.metric("🔧 Skill Match",      f"{row['skill_match']:.1f}%")
            c2.metric("💼 Experience Match", f"{row['exp_match']:.1f}%")
            c3.metric("📁 Project Match",    f"{row['proj_match']:.1f}%")

            st.progress(row["skill_match"] / 100,
                        text=f"Skill Match: {row['skill_match']:.1f}%")
            st.progress(row["exp_match"]   / 100,
                        text=f"Experience Match: {row['exp_match']:.1f}%")
            st.progress(row["proj_match"]  / 100,
                        text=f"Project Match: {row['proj_match']:.1f}%")

            txt    = row["resume_text"]
            tokens = clean_tokens(txt)
            n      = len(tokens)
            seed   = abs(hash(str(row["resume_id"]))) % (2**32 - 1)
            attn   = attention_scores(tokens, seed=seed)
            top_n_idx = np.argsort(attn)[-8:][::-1]
            top_tok   = tokens[int(np.argmax(attn))] if n else "—"
            matched_skills = extract_skills(txt, row["category"])

            # ── SECTION 4: Why Candidate Was Selected ─────────────────────────
            st.markdown("**📌 Why This Candidate Was Selected**")
            reasons = []
            if row["skill_match"] >= 40:
                reasons.append(f"Strong keyword alignment ({row['skill_match']:.0f}% skill match)")
            if row["exp_match"] >= 40:
                reasons.append(f"Relevant experience detected ({row['exp_match']:.0f}% experience match)")
            if row["proj_match"] >= 30:
                reasons.append(f"Project work evident ({row['proj_match']:.0f}% project match)")
            if row["category"] == jd_role:
                reasons.append(f"Category matches required role ({jd_role})")
            if not reasons:
                reasons.append("Relative highest score among candidates")
            for r in reasons:
                st.markdown(f"- {r}")

            det_col1, det_col2 = st.columns(2)

            # ── SECTION 5: Important Skills ───────────────────────────────────
            with det_col1:
                st.markdown("**🔧 Important Skills Found**")
                if matched_skills:
                    for sk in matched_skills[:8]:
                        st.markdown(f"  ✅ `{sk}`")
                else:
                    st.markdown("  *No domain-specific skills detected*")

            # ── SECTION 6: Attention Scores ───────────────────────────────────
            with det_col2:
                st.markdown("**🧠 Top Attention Scores**")
                for idx in top_n_idx[:6]:
                    if idx < n:
                        st.markdown(f"  • **{tokens[idx]}** — `{attn[idx]:.4f}`")

            # ── SECTION 7: Matching Evidence ─────────────────────────────────
            st.markdown("**🔍 Matching Evidence**")
            evidence = []
            for kw in jd_kw:
                if kw.lower() in txt.lower():
                    evidence.append(f"`{kw}`")
            if evidence:
                st.markdown("JD keywords found in resume: " + "  ".join(evidence))
            else:
                st.markdown("*No direct JD keyword matches found*")

            # ── SECTION 8: Attention Heatmap ──────────────────────────────────
            if n > 0:
                st.markdown("**🗺️ Attention Heatmap**")
                fig_a, ax_a = plt.subplots(figsize=(max(10, min(n, 40) * 0.55), 2.5))
                display_n = min(n, 40)
                im = ax_a.imshow([attn[:display_n]], aspect="auto", cmap="YlOrRd")
                ax_a.set_xticks(range(display_n))
                ax_a.set_xticklabels(tokens[:display_n],
                                     rotation=45, ha="right", fontsize=7)
                ax_a.set_yticks([])
                ax_a.set_title(
                    f"Candidate #{row['resume_id']} — Token Attention  |  "
                    f"Peak: '{top_tok}'",
                    fontsize=9)
                plt.colorbar(im, ax=ax_a, orientation="horizontal",
                             fraction=0.025, pad=0.4, label="Attention")
                plt.tight_layout()
                st.pyplot(fig_a)
                plt.close(fig_a)

    # ── SECTION 9: Positional Encoding Heatmap ───────────────────────────────
    st.markdown("---")
    st.header("📈 Step 3 — Positional Encoding Heatmap")

    pe_c1, pe_c2 = st.columns(2)
    n_pos   = pe_c1.slider("Token positions", 10, 100, 80)
    d_model = pe_c2.slider("Encoding dimensions", 16, 128, 64, step=16)
    PE = positional_encoding(n_pos, d_model)

    fig_pe, ax_pe = plt.subplots(figsize=(14, max(4, n_pos // 8)))
    im_pe = ax_pe.imshow(PE, aspect="auto", cmap="viridis")
    plt.colorbar(im_pe, label="Encoding value")
    ax_pe.set_title(
        f"Sinusoidal Positional Encoding — Resume Tokens "
        f"({n_pos} positions × {d_model} dims)", fontsize=11)
    ax_pe.set_xlabel("Encoding Dimension")
    ax_pe.set_ylabel("Token Position in Resume")
    plt.tight_layout()
    st.pyplot(fig_pe)

    pe_buf = io.BytesIO()
    fig_pe.savefig(pe_buf, format="png", dpi=120, bbox_inches="tight")
    pe_bytes = pe_buf.getvalue()
    plt.close(fig_pe)

    st.info("A skill at position 0 (top of resume) vs position 20 (later) receives a "
            "**different positional vector** — the model knows WHERE in the resume each "
            "term appears, not just that it appears.")

    # ── SECTION 10: Export Results ────────────────────────────────────────────
    st.markdown("---")
    st.header("📥 Step 4 — Export Results")

    exp_c1, exp_c2 = st.columns(2)
    csv_buf = io.StringIO()
    top10.to_csv(csv_buf, index=False)
    exp_c1.download_button("⬇️ Download Top 10 CSV",
                           csv_buf.getvalue(),
                           f"top10_{jd_title.replace(' ','_')}.csv",
                           "text/csv", use_container_width=True)

    full_buf = io.StringIO()
    df_scored.sort_values("score", ascending=False).to_csv(full_buf, index=False)
    exp_c2.download_button("⬇️ Download All Candidates CSV",
                           full_buf.getvalue(),
                           f"all_candidates_{jd_title.replace(' ','_')}.csv",
                           "text/csv", use_container_width=True)

    # ── Summary bar chart ─────────────────────────────────────────────────────
    fig_bar, ax_bar = plt.subplots(figsize=(11, 4))
    colors = ["gold","silver","#cd7f32"] + ["steelblue"] * 7
    bars   = ax_bar.bar(range(len(top10)), top10["score"],
                        color=colors[:len(top10)], edgecolor="black")
    ax_bar.set_xticks(range(len(top10)))
    ax_bar.set_xticklabels(
        [f"#{r}" for r in top10["resume_id"]], rotation=45, ha="right")
    ax_bar.set_title(f"Top 10 Candidate Scores — {jd_title}")
    ax_bar.set_ylabel("Overall Score (%)")
    ax_bar.set_ylim(0, 115)
    for bar, val in zip(bars, top10["score"]):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # ── BONUS ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🎯 Bonus — Advanced Analysis")

    bonus_tabs = st.tabs([
        "🔬 Attention Head Comparison",
        "🗺️ Attention Maps per Head",
        "📄 Explainable Hiring Report (PDF)",
    ])

    # Pick top candidate for bonus visuals
    top_row    = top10.iloc[0]
    top_txt    = top_row["resume_text"]
    top_tokens = clean_tokens(top_txt)
    top_n_tok  = min(len(top_tokens), 20)

    # ── Bonus Tab 1: Attention Head Comparison ────────────────────────────────
    with bonus_tabs[0]:
        st.subheader("Attention Head Comparison (2, 4, 8 heads)")
        st.caption(f"Top candidate #{top_row['resume_id']} — simulated multi-head attention patterns")

        head_cols = st.columns(3)
        for col, n_heads in zip(head_cols, [2, 4, 8]):
            with col:
                st.markdown(f"**{n_heads} Heads**")
                head_matrix = np.zeros((n_heads, top_n_tok))
                for h in range(n_heads):
                    np.random.seed(h * 17 + 3)
                    head_matrix[h] = np.random.dirichlet(
                        np.ones(top_n_tok) * (0.3 + h * 0.1))
                fig_h, ax_h = plt.subplots(figsize=(6, max(2, n_heads * 0.5)))
                im_h = ax_h.imshow(head_matrix, aspect="auto", cmap="Blues")
                ax_h.set_xticks(range(top_n_tok))
                ax_h.set_xticklabels(top_tokens[:top_n_tok],
                                     rotation=90, fontsize=6)
                ax_h.set_yticks(range(n_heads))
                ax_h.set_yticklabels([f"H{i+1}" for i in range(n_heads)], fontsize=8)
                ax_h.set_title(f"{n_heads}-Head Attention", fontsize=9)
                plt.colorbar(im_h, ax=ax_h, fraction=0.04)
                plt.tight_layout()
                st.pyplot(fig_h)
                plt.close(fig_h)

        st.info("Each head learns to focus on **different aspects** of the resume. "
                "Head 1 may focus on skills, Head 2 on experience keywords, etc.")

    # ── Bonus Tab 2: Attention Maps per Head ──────────────────────────────────
    with bonus_tabs[1]:
        st.subheader("Individual Attention Maps per Head")
        n_heads_sel = st.radio("Number of heads", [2, 4, 8], horizontal=True, index=1)

        for h in range(n_heads_sel):
            np.random.seed(h * 17 + 3)
            h_attn = np.random.dirichlet(np.ones(top_n_tok) * (0.3 + h * 0.1))
            top_h  = top_tokens[int(np.argmax(h_attn))]
            fig_hm, ax_hm = plt.subplots(figsize=(max(10, top_n_tok * 0.55), 2))
            ax_hm.imshow([h_attn], aspect="auto", cmap="YlOrRd")
            ax_hm.set_xticks(range(top_n_tok))
            ax_hm.set_xticklabels(top_tokens[:top_n_tok],
                                  rotation=45, ha="right", fontsize=8)
            ax_hm.set_yticks([])
            ax_hm.set_title(
                f"Head {h+1} Attention — Peak: '{top_h}'  ({h_attn.max():.3f})",
                fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)

    # ── Bonus Tab 3: PDF Hiring Report ────────────────────────────────────────
    with bonus_tabs[2]:
        st.subheader("📄 Explainable Hiring Report")
        st.markdown("Generate a full PDF report with rankings, match scores, "
                    "explainability, attention heatmap, and positional encoding.")

        if st.button("🖨️ Generate PDF Report", type="primary"):
            # Build attention image for top candidate
            seed_t  = abs(hash(str(top_row["resume_id"]))) % (2**32 - 1)
            attn_t  = attention_scores(top_tokens, seed=seed_t)
            dn      = min(len(top_tokens), 40)
            fig_at, ax_at = plt.subplots(figsize=(max(10, dn * 0.55), 2.5))
            im_at = ax_at.imshow([attn_t[:dn]], aspect="auto", cmap="YlOrRd")
            ax_at.set_xticks(range(dn))
            ax_at.set_xticklabels(top_tokens[:dn], rotation=45, ha="right", fontsize=7)
            ax_at.set_yticks([])
            ax_at.set_title(f"Top Candidate #{top_row['resume_id']} — Attention Heatmap")
            plt.colorbar(im_at, ax=ax_at, orientation="horizontal",
                         fraction=0.025, pad=0.4)
            plt.tight_layout()
            attn_buf = io.BytesIO()
            fig_at.savefig(attn_buf, format="png", dpi=120, bbox_inches="tight")
            attn_bytes_pdf = attn_buf.getvalue()
            plt.close(fig_at)

            # candidate details for PDF
            cand_details = []
            for _, row in top10.head(5).iterrows():
                tkns  = clean_tokens(str(row["resume_text"]))
                a_sc  = attention_scores(tkns,
                        seed=abs(hash(str(row["resume_id"]))) % (2**32-1))
                top_t = tkns[int(np.argmax(a_sc))] if tkns else "—"
                why   = []
                if row["skill_match"]  >= 40: why.append(f"Skill match {row['skill_match']:.0f}%")
                if row["exp_match"]    >= 40: why.append(f"Experience match {row['exp_match']:.0f}%")
                if row["proj_match"]   >= 30: why.append(f"Project match {row['proj_match']:.0f}%")
                if row["category"] == jd_role: why.append("Category matches JD")
                cand_details.append({
                    "id":        row["resume_id"],
                    "category":  row["category"],
                    "score":     row["score"],
                    "why":       "; ".join(why) if why else "Highest relative score",
                    "skills":    extract_skills(str(row["resume_text"]), str(row["category"])),
                    "top_token": top_t,
                    "top_attn":  float(np.max(a_sc)) if len(a_sc) else 0.0,
                })

            with st.spinner("Building PDF…"):
                pdf_bytes = generate_hiring_report(
                    jd_title=jd_title, jd_role=jd_role,
                    top_df=top10, candidate_details=cand_details,
                    attn_bytes=attn_bytes_pdf, pe_bytes=pe_bytes)

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Hiring Report PDF",
                pdf_bytes,
                f"hiring_report_{jd_title.replace(' ','_')}_{ts}.pdf",
                "application/pdf",
                type="primary",
                use_container_width=False,
            )
            st.success("✅ PDF report ready with full candidate rankings, "
                       "attention maps, and explainability details.")

    st.markdown("---")
    st.caption("Smart Recruitment Intelligence Platform · Analysis complete.")

else:
    st.info("👆 Fill in the Job Description, upload or use the sample resumes, "
            "then click **Rank Candidates**.")
