# ==============================================================
# Sistem Rekomendasi Penyuntingan Kata pada Berita Universitas
# Algoritma : Jaro-Winkler + IndoBERT (Fine-tuning) + Hybrid
# Interface  : Streamlit
# Penulis    : Noeni Indah Sulistiyani
# ==============================================================

import os
import re
import time
import unicodedata
import tempfile

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from config import DRIVE_IDS, MODEL_CFG, JW_CFG

# ==============================================================
# KONFIGURASI HALAMAN
# ==============================================================

st.set_page_config(
    page_title="Sistem Penyuntingan Kata Berita UIN",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# UTILITAS TEKS
# ==============================================================

def normalize_unicode(text: str) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFKC", str(text))

def clean_whitespace(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\u200b", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()

def normalize_token(token: str) -> str:
    if pd.isna(token):
        return ""
    return clean_whitespace(normalize_unicode(token)).lower().strip()

def simple_sentence_split(text: str) -> list:
    text = clean_whitespace(str(text))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def simple_tokenize(sentence: str) -> list:
    sentence = re.sub(r"[^\w\s]", " ", sentence)
    return [t.strip() for t in sentence.split() if t.strip()]

# ==============================================================
# PATH LOKAL UNTUK CACHE UNDUHAN
# ==============================================================

_TMP           = tempfile.gettempdir()
MODEL_LOCAL    = os.path.join(_TMP, "model_indobert_best")
LEXICON_LOCAL  = os.path.join(_TMP, "leksikon")
os.makedirs(MODEL_LOCAL,   exist_ok=True)
os.makedirs(LEXICON_LOCAL, exist_ok=True)

LEXICON_COL_MAP = {
    "kbbi"              : "kata",
    "kata_inggris"      : "headword",
    "akronim"           : "akronim",
    "daftar_lembaga"    : "Nama Lembaga",
    "daftar_nama_orang" : "Nama",
    "istilah_islam"     : "Kata",
}

# ==============================================================
# LOAD RESOURCES (cache — hanya dijalankan sekali)
# ==============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Unduh (jika belum ada) dan load model IndoBERT dari Drive."""
    config_path = os.path.join(MODEL_LOCAL, "config.json")
    if not os.path.exists(config_path):
        gdown.download_folder(
            id=DRIVE_IDS["model_indobert"],
            output=MODEL_LOCAL,
            quiet=False,
            use_cookies=False,
        )
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_LOCAL)
    model     = BertForSequenceClassification.from_pretrained(
        MODEL_LOCAL,
        num_labels=MODEL_CFG["num_labels"],
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_lexicons():
    """Unduh (jika belum ada) dan bangun semua set leksikon dari Drive."""

    def read_csv_safe(path: str) -> pd.DataFrame:
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception:
                continue
        return pd.DataFrame()

    def to_set(df: pd.DataFrame, col: str) -> set:
        if col not in df.columns:
            col = df.columns[0] if len(df.columns) > 0 else None
        if col is None:
            return set()
        return set(df[col].dropna().astype(str).str.strip().str.lower())

    # Unduh semua file leksikon
    lex_dfs = {}
    for key in ["kbbi", "kata_inggris", "kata_serapan",
                "akronim", "daftar_lembaga",
                "daftar_nama_orang", "istilah_islam"]:
        local_path = os.path.join(LEXICON_LOCAL, f"{key}.csv")
        if not os.path.exists(local_path):
            gdown.download(
                id=DRIVE_IDS[key],
                output=local_path,
                quiet=True,
            )
        lex_dfs[key] = read_csv_safe(local_path)

    # Bangun set
    kbbi_set    = to_set(lex_dfs["kbbi"], "kata")
    inggris_set = to_set(lex_dfs["kata_inggris"], "headword") - kbbi_set

    whitelist_set = set()
    for key in ["akronim", "daftar_lembaga",
                "daftar_nama_orang", "istilah_islam"]:
        col = LEXICON_COL_MAP[key]
        whitelist_set.update(to_set(lex_dfs[key], col))

    # Kata serapan: pasangan asal → serapan
    serapan_map = {}
    serapan_set = set()
    df_s = lex_dfs.get("kata_serapan", pd.DataFrame())
    if not df_s.empty:
        col_asal = next(
            (c for c in df_s.columns
             if "asal" in c.lower() or "asing" in c.lower()),
            df_s.columns[0]
        )
        col_serapan = next(
            (c for c in df_s.columns
             if "serapan" in c.lower() or "hasil" in c.lower()),
            df_s.columns[-1]
        )
        for _, row in df_s.iterrows():
            asal    = normalize_token(str(row[col_asal]))
            serapan = normalize_token(str(row[col_serapan]))
            if asal and serapan:
                serapan_map[asal] = serapan
                serapan_set.add(asal)

    kbbi_list = sorted(kbbi_set)
    return (kbbi_set, inggris_set, whitelist_set,
            serapan_map, serapan_set, kbbi_list)

# ==============================================================
# ALGORITMA JARO-WINKLER
# ==============================================================

def jaro_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    match_dist = max(0, max(len1, len2) // 2 - 1)
    s1m = [False] * len1
    s2m = [False] * len2
    matches = transpositions = 0
    for i in range(len1):
        for j in range(max(0, i - match_dist),
                       min(i + match_dist + 1, len2)):
            if s2m[j] or s1[i] != s2[j]:
                continue
            s1m[i] = s2m[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not s1m[i]:
            continue
        while not s2m[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    return (matches/len1 + matches/len2 +
            (matches - transpositions/2)/matches) / 3


def jaro_winkler_similarity(s1: str, s2: str,
                             p: float = 0.1) -> float:
    jaro = jaro_similarity(s1, s2)
    prefix = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


def classify_token(token: str, kbbi_set, inggris_set,
                   whitelist_set, serapan_set) -> str:
    t = normalize_token(token)
    if not t:
        return "KOSONG"
    if t in whitelist_set:
        return "WHITELIST_KHUSUS"
    if t in serapan_set:
        return "KATA_SERAPAN"
    if t in kbbi_set:
        return "KBBI_VALID"
    if t in inggris_set:
        return "KATA_INGGRIS"
    return "TIDAK_DIKENAL"


def predict_jw(token: str, kbbi_set, inggris_set,
               whitelist_set, serapan_set, kbbi_list,
               threshold: float, top_k: int = 5) -> dict:
    t      = normalize_token(token)
    status = classify_token(t, kbbi_set, inggris_set,
                             whitelist_set, serapan_set)

    if status in ("WHITELIST_KHUSUS", "KBBI_VALID", "KATA_SERAPAN", "KOSONG"):
        return {"pred": 0, "max_sim": 1.0,
                "best_match": t, "top_k_recs": [], "status": status}

    sims = [(w, jaro_winkler_similarity(t, w)) for w in kbbi_list]
    sims.sort(key=lambda x: x[1], reverse=True)
    best_word, best_sim = sims[0]
    pred = 0 if best_sim >= threshold else 1

    return {
        "pred"      : pred,
        "max_sim"   : round(best_sim, 4),
        "best_match": best_word,
        "top_k_recs": [w for w, _ in sims[:top_k]],
        "status"    : status,
    }

# ==============================================================
# PREDIKSI INDOBERT
# ==============================================================

def predict_bert(kalimat: str, token: str,
                 tokenizer, model, device) -> dict:
    text_a = clean_whitespace(kalimat)
    text_b = normalize_token(token)
    enc    = tokenizer(
        text_a, text_b,
        max_length=MODEL_CFG["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device),
            token_type_ids = enc["token_type_ids"].to(device),
        )
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    return {
        "pred"        : pred,
        "prob_correct": round(float(probs[0]), 4),
        "prob_error"  : round(float(probs[1]), 4),
    }

# ==============================================================
# PIPELINE ANALISIS TEKS
# ==============================================================

def analyze_text(text: str, model_choice: str,
                 tokenizer, bert_model, device,
                 kbbi_set, inggris_set, whitelist_set,
                 serapan_map, serapan_set, kbbi_list) -> list:
    sentences = simple_sentence_split(text)
    results   = []

    for sent in sentences:
        tokens = simple_tokenize(sent)
        for tok in tokens:
            t = normalize_token(tok)
            if not t or len(t) < 2:
                continue

            jw_res   = predict_jw(
                t, kbbi_set, inggris_set,
                whitelist_set, serapan_set, kbbi_list,
                threshold=JW_CFG["threshold"],
                top_k=JW_CFG["top_k"],
            )
            bert_res = predict_bert(
                sent, t, tokenizer, bert_model, device
            )

            jw_pred   = jw_res["pred"]
            bert_pred = bert_res["pred"]

            if model_choice == "Jaro-Winkler":
                final_pred = jw_pred
            elif model_choice == "IndoBERT":
                final_pred = bert_pred
            else:  # Hybrid-OR
                final_pred = 1 if (jw_pred == 1 or bert_pred == 1) else 0

            if final_pred == 0:
                continue

            # Tipe error
            if jw_pred == 1 and bert_pred == 0:
                tipe = "TYPO (leksikal)"
            elif jw_pred == 0 and bert_pred == 1:
                tipe = "Real-Word Error (kontekstual)"
            else:
                tipe = "TYPO + Kontekstual"

            # Rekomendasi
            recs    = jw_res["top_k_recs"]
            catatan = ""
            if jw_res["status"] == "KATA_INGGRIS":
                padanan = serapan_map.get(t)
                catatan = (f"Padanan KBBI: '{padanan}'"
                           if padanan
                           else "Gunakan huruf miring jika dipertahankan")

            results.append({
                "token"         : tok,
                "kalimat"       : sent,
                "tipe_error"    : tipe,
                "jw_pred"       : "ERROR" if jw_pred else "OK",
                "bert_pred"     : "ERROR" if bert_pred else "OK",
                "prob_error"    : bert_res["prob_error"],
                "jw_sim"        : jw_res["max_sim"],
                "rekomendasi"   : recs,
                "catatan"       : catatan,
            })

    return results

# ==============================================================
# ANTARMUKA STREAMLIT
# ==============================================================

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📝 Sistem Penyuntingan Kata")
    st.markdown("Berita UIN Jakarta · 2026")
    st.markdown("---")

    model_choice = st.selectbox(
        "Pilih Model Deteksi",
        options=["Hybrid-OR", "IndoBERT", "Jaro-Winkler"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Performa Model (test set)**")

    perf = {
        "Hybrid-OR"    : {"F1": "0.9976", "Recall": "0.9988", "Precision": "0.9964"},
        "IndoBERT"     : {"F1": "0.9952", "Recall": "0.9940", "Precision": "0.9964"},
        "Jaro-Winkler" : {"F1": "0.9824", "Recall": "0.9654", "Precision": "1.0000"},
    }
    for metric, val in perf[model_choice].items():
        st.metric(metric, val)

    st.markdown("---")
    st.caption("Noeni Indah Sulistiyani\nTeknik Informatika · UIN Jakarta")

# ── Header ────────────────────────────────────────────────────
st.title("📝 Sistem Rekomendasi Penyuntingan Kata")
st.markdown(
    "Deteksi kesalahan penulisan pada teks berita universitas "
    "menggunakan **Jaro-Winkler** dan **IndoBERT**."
)
st.markdown("---")

# ── Load resources ────────────────────────────────────────────
with st.spinner("Memuat model dan leksikon..."):
    tokenizer, bert_model, device = load_model()
    (kbbi_set, inggris_set, whitelist_set,
     serapan_map, serapan_set, kbbi_list) = load_lexicons()

st.success(f"Model **{model_choice}** siap digunakan.", icon="✅")

# ── Tab input ─────────────────────────────────────────────────
tab_teks, tab_file = st.tabs(["✏️ Input Teks", "📂 Upload File"])

with tab_teks:
    input_text = st.text_area(
        "Masukkan teks berita:",
        height=180,
        placeholder=(
            "Contoh: Rektor UIN Jakarta menyambut positif pencpaaian ini. "
            "Menurutnya capaian ini merupakan bagian dari upaya berkelanjutan "
            "universitas dalam memperkuat kualitas academic di tingkat global."
        ),
    )
    run_teks = st.button("🔍 Analisis", type="primary",
                          use_container_width=True, key="btn_teks")

with tab_file:
    uploaded = st.file_uploader(
        "Upload file berita (.txt atau .docx)", type=["txt", "docx"]
    )
    file_text    = ""
    run_file     = False
    if uploaded:
        if uploaded.name.endswith(".txt"):
            file_text = uploaded.read().decode("utf-8", errors="replace")
        else:
            import docx as _docx
            doc       = _docx.Document(uploaded)
            file_text = "\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            )
        st.text_area("Isi file:", value=file_text,
                     height=180, disabled=True)
        run_file = st.button("🔍 Analisis File", type="primary",
                              use_container_width=True, key="btn_file")

# ── Jalankan analisis ─────────────────────────────────────────
text_to_run = ""
if run_teks and input_text.strip():
    text_to_run = input_text
elif run_file and file_text.strip():
    text_to_run = file_text

if text_to_run:
    with st.spinner(f"Menganalisis dengan {model_choice}..."):
        t0      = time.time()
        results = analyze_text(
            text_to_run, model_choice,
            tokenizer, bert_model, device,
            kbbi_set, inggris_set, whitelist_set,
            serapan_map, serapan_set, kbbi_list,
        )
        elapsed = round(time.time() - t0, 2)

    st.markdown("---")

    # Metrik ringkasan
    total_tok = len([t for t in simple_tokenize(
        re.sub(r"[^\w\s]", " ", text_to_run)) if len(t) >= 2])
    n_err     = len(results)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Token Diperiksa", total_tok)
    c2.metric("Token Bermasalah",      n_err)
    c3.metric("Waktu Analisis",        f"{elapsed}s")

    if n_err == 0:
        st.success("✅ Tidak ditemukan kesalahan penulisan.", icon="✅")
    else:
        st.warning(f"⚠️ Ditemukan **{n_err} token** yang perlu diperiksa.")

        # Tabel ringkasan
        st.markdown("### Tabel Hasil Deteksi")
        tabel = pd.DataFrame([{
            "Token"         : r["token"],
            "Tipe Error"    : r["tipe_error"],
            "JW"            : r["jw_pred"],
            "BERT"          : r["bert_pred"],
            "Skor JW"       : r["jw_sim"],
            "Prob Error BERT": r["prob_error"],
            "Rekomendasi"   : ", ".join(r["rekomendasi"]) if r["rekomendasi"] else "-",
        } for r in results])

        st.dataframe(
            tabel,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Skor JW"        : st.column_config.NumberColumn(format="%.4f"),
                "Prob Error BERT": st.column_config.ProgressColumn(
                    format="%.4f", min_value=0, max_value=1
                ),
            },
        )

        # Detail per token
        st.markdown("### Detail Per Token")
        for r in results:
            icon = "🔴" if "TYPO" in r["tipe_error"] else "🟡"
            with st.expander(f"{icon} **{r['token']}** — {r['tipe_error']}"):
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown(f"**Token:** `{r['token']}`")
                    st.markdown(f"**Tipe:** {r['tipe_error']}")
                    st.markdown(
                        f"**Jaro-Winkler:** {r['jw_pred']} "
                        f"(sim = {r['jw_sim']})"
                    )
                    st.markdown(
                        f"**IndoBERT:** {r['bert_pred']} "
                        f"(prob error = {r['prob_error']})"
                    )
                    if r["catatan"]:
                        st.info(r["catatan"])

                with col_r:
                    st.markdown("**Rekomendasi kata (top-5):**")
                    if r["rekomendasi"]:
                        for rec in r["rekomendasi"]:
                            st.code(rec)
                    else:
                        st.caption("Tidak ada rekomendasi spesifik.")

                st.markdown("**Kalimat konteks:**")
                highlighted = re.sub(
                    rf"\b{re.escape(r['token'])}\b",
                    f"**:red[{r['token']}]**",
                    r["kalimat"],
                )
                st.markdown(f"> {highlighted}")
