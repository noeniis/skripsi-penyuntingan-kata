
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
# CSS KUSTOM — Highlight + Tooltip
# ==============================================================

st.markdown("""
<style>
/* ── Font & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Highlighted word wrapper ── */
.word-wrap {
    position: relative;
    display: inline-block;
    margin: 0 1px;
}

/* ── Base token badge style ── */
.token-badge {
    display: inline-block;
    padding: 2px 5px;
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
    font-size: 1em;
    font-weight: 600;
    cursor: help;
    border-bottom: 2.5px solid transparent;
    transition: filter 0.15s;
}
.token-badge:hover {
    filter: brightness(0.88);
}

/* ── Kategori warna ── */
.flag-typo          { background: #ffe0e0; color: #b91c1c; border-bottom-color: #b91c1c; }
.flag-real-word     { background: #fff3cd; color: #92400e; border-bottom-color: #d97706; }
.flag-typo-konteks  { background: #fce7f3; color: #9d174d; border-bottom-color: #db2777; }
.flag-kata-inggris  { background: #dbeafe; color: #1e40af; border-bottom-color: #3b82f6; }
.flag-kata-serapan  { background: #d1fae5; color: #065f46; border-bottom-color: #10b981; }
.flag-whitelist     { background: #ede9fe; color: #4c1d95; border-bottom-color: #7c3aed; }

/* ── Tooltip / Panel klik ── */
.word-wrap {
    position: relative;
    display: inline-block;
    margin: 0 1px;
}

/* Panel detail yang muncul saat diklik */
.token-panel {
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 12px;
    padding: 16px 20px;
    width: 300px;
    font-size: 0.85em;
    font-family: 'DM Sans', sans-serif;
    line-height: 1.6;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    z-index: 99999;
    white-space: normal;
}
.token-panel.active { display: block; }

.panel-close {
    float: right;
    cursor: pointer;
    font-size: 1.1em;
    color: #f38ba8;
    margin-left: 8px;
}
.panel-close:hover { color: #ff5555; }

.panel-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.3);
    z-index: 99998;
}
.panel-overlay.active { display: block; }

.rec-btn {
    display: inline-block;
    margin: 3px 4px 3px 0;
    padding: 3px 10px;
    background: #313244;
    color: #cba6f7;
    border: 1px solid #7c3aed;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.88em;
    font-family: 'DM Sans', sans-serif;
    transition: background 0.15s;
}
.rec-btn:hover { background: #45475a; }

.ignore-btn {
    display: inline-block;
    margin-top: 8px;
    padding: 4px 12px;
    background: #1e1e2e;
    color: #a6e3a1;
    border: 1px solid #40a02b;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85em;
    font-family: 'DM Sans', sans-serif;
    transition: background 0.15s;
}
.ignore-btn:hover { background: #2a2a3e; }

.panel-divider {
    border: none;
    border-top: 1px solid #45475a;
    margin: 8px 0;
}

/* Tanda bahwa token sudah diabaikan */
.token-ignored {
    text-decoration: line-through;
    opacity: 0.45;
    cursor: default;
}

/* ── Teks normal (tidak bermasalah) ── */
.token-normal {
    font-family: 'DM Sans', sans-serif;
    font-size: 1em;
}

/* ── Kontainer teks hasil ── */
.text-preview-box {
    background: #fafafa;
    border: 1.5px solid #e5e7eb;
    border-radius: 10px;
    padding: 20px 24px;
    line-height: 2.2em;
    font-family: 'Lora', serif;
    font-size: 1.05em;
    color: #111827;
    word-spacing: 1px;
}

/* ── Legenda warna ── */
.legend-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 18px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85em;
    font-family: 'DM Sans', sans-serif;
}
.legend-dot {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    flex-shrink: 0;
}

/* ── Tabel ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)

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

# ==============================================================
# STEMMING RINGAN — Deteksi Kata Berimbuhan
# ==============================================================
#
# Tujuan: jika token tidak ditemukan di KBBI, coba lepas afiks
# umum bahasa Indonesia dan cek bentuk dasarnya.
# Ini bukan full Nazief-Adriani/PySastrawi stemmer, melainkan
# rule-based stripping afiks paling produktif di teks berita.
#
# Justifikasi: pendekatan strip-afiks sederhana untuk spell check
# sudah digunakan di banyak sistem (Hunspell, Aspell) dan terbukti
# efektif mengurangi false positive pada bahasa berimbuhan tinggi
# (Adriani et al., 2007 — Confix-Stripping Stemmer for Indonesian).
#
# Urutan stripping: sufiks → prefiks → konfiks (kombinasi)
# ==============================================================

_SUFIKS  = ["nya", "kan", "lah", "kah", "pun", "ku", "mu", "an", "i"]
_PREFIKS = ["me", "mem", "men", "meng", "meny", "me",
            "ber", "be", "ter", "pe", "per", "di", "ke", "se"]
_KONFIKS = [("me","kan"), ("me","i"), ("ber","kan"), ("ber","i"),
            ("per","kan"), ("per","i"), ("di","kan"), ("di","i"),
            ("ke","an"), ("pe","an"), ("per","an")]

def strip_afiks(token: str) -> list:
    """
    Kembalikan kandidat bentuk dasar setelah stripping afiks.
    Hanya stripping yang menghasilkan panjang >= 3 karakter.
    """
    candidates = set()
    t = token

    # Strip sufiks
    for suf in _SUFIKS:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            candidates.add(t[: -len(suf)])

    # Strip prefiks dari token asli dan dari hasil strip sufiks
    sources = {t} | candidates.copy()
    for src in sources:
        for pre in _PREFIKS:
            if src.startswith(pre) and len(src) - len(pre) >= 3:
                candidates.add(src[len(pre):])

    # Strip konfiks (prefiks + sufiks sekaligus)
    for pre, suf in _KONFIKS:
        if t.startswith(pre) and t.endswith(suf):
            base = t[len(pre): -len(suf)]
            if len(base) >= 3:
                candidates.add(base)

    return list(candidates)


def is_kata_berimbuhan_valid(token: str, kbbi_set: set) -> bool:
    """
    True jika token adalah kata berimbuhan yang bentuk dasarnya ada di KBBI.
    Langsung return False jika token sudah ada di KBBI (tidak perlu cek).
    """
    if token in kbbi_set:
        return True
    for base in strip_afiks(token):
        if base in kbbi_set:
            return True
    return False


def simple_sentence_split(text: str) -> list:
    text = clean_whitespace(str(text))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def simple_tokenize(sentence: str) -> list:
    sentence = re.sub(r"[^\w\s]", " ", sentence)
    return [t.strip() for t in sentence.split() if t.strip()]

# ==============================================================
# PRE-PROCESSING FILTER — Reduksi False Positive
# ==============================================================
#
# Filter ini diterapkan SEBELUM token masuk ke model deteksi.
# Tidak mengubah parameter model (threshold, strategi hybrid).
# Justifikasi ilmiah:
#   [F1] Gelar akademik  → pre-processing domain teks akademik
#        (Kukich, 1992; Hunspell documentation)
#   [F2] Token < 3 kar.  → standar minimum length di spell checker
#        (Norvig, 2009; GNU Aspell)
#   [F3] Semua huruf kap → kemungkinan akronim/singkatan
#        (Jurafsky & Martin, 2023, Ch. 2)
#   [F4] Proper noun heur.→ token berawalan huruf kapital di tengah
#        kalimat dianggap Named Entity, dikecualikan dari leksikon
#        (Jurafsky & Martin, 2023, Ch. 8 — NER heuristic)
# ==============================================================

# [F1] Pola gelar akademik Indonesia yang umum di teks berita UIN
_GELAR_RE = re.compile(
    r"^("
    r"prof|dr|ir|hj?|drs?|"
    r"s\.pd|s\.t|s\.kom|s\.ag|s\.sos|s\.h|s\.e|s\.i\.kom|s\.psi|"
    r"m\.pd|m\.si|m\.kom|m\.ag|m\.h|m\.e|m\.a|m\.sc|m\.eng|"
    r"ph\.d|d\.sc|apt|lc|s\.farm|s\.kep|s\.ked|"
    r"sp|spd|sag|sh|se|st|skom|mpd|msi|mkom|mag|mh|me|ma|msc|phd"
    r")$",
    re.IGNORECASE,
)

def is_gelar_akademik(token: str) -> bool:
    """[F1] True jika token adalah gelar akademik."""
    return bool(_GELAR_RE.match(token.strip(".")))

def is_token_terlalu_pendek(token: str, min_len: int = 3) -> bool:
    """[F2] True jika token lebih pendek dari batas minimum."""
    return len(token) < min_len

def is_akronim_kapital(token: str) -> bool:
    """[F3] True jika token seluruhnya huruf kapital (kemungkinan akronim)."""
    return token.isupper() and len(token) >= 2

def is_proper_noun_heuristic(token: str, position: int) -> bool:
    """
    [F4] True jika token kemungkinan Named Entity berdasarkan heuristik
    kapitalisasi. Token di posisi > 0 (bukan awal kalimat) yang diawali
    huruf kapital dianggap nama orang/tempat/institusi.
    Posisi 0 dikecualikan karena huruf kapital di awal kalimat adalah
    aturan tata bahasa, bukan indikator Named Entity.
    """
    return position > 0 and len(token) > 0 and token[0].isupper()

def should_skip_token(token: str, position: int,
                      skip_proper_noun: bool = True) -> tuple:
    """
    Kembalikan (True, alasan) jika token harus dilewati,
    (False, "") jika token perlu dicek.
    """
    if is_gelar_akademik(token):
        return True, "gelar_akademik"
    if is_token_terlalu_pendek(token):
        return True, "terlalu_pendek"
    if is_akronim_kapital(token):
        return True, "akronim_kapital"
    if skip_proper_noun and is_proper_noun_heuristic(token, position):
        return True, "proper_noun"
    return False, ""

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

        vals = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return set(v for v in vals if len(v) >= 2)

    # ==========================================================
    # DOWNLOAD SEMUA LEKSIKON
    # ==========================================================

    lex_dfs = {}

    for key in [
        "kbbi",
        "kata_inggris",
        "kata_serapan",
        "akronim",
        "daftar_lembaga",
        "daftar_nama_orang",
        "istilah_islam",
        "sample_correct_2025",   # <<< TAMBAHAN BARU
    ]:

        local_path = os.path.join(LEXICON_LOCAL, f"{key}.csv")

        if not os.path.exists(local_path):
            gdown.download(
                id=DRIVE_IDS[key],
                output=local_path,
                quiet=True,
            )

        lex_dfs[key] = read_csv_safe(local_path)

    # ==========================================================
    # KBBI
    # ==========================================================

    kbbi_set = to_set(
        lex_dfs["kbbi"],
        "kata"
    )

    # ==========================================================
    # INGGRIS
    # ==========================================================

    inggris_set = (
        to_set(
            lex_dfs["kata_inggris"],
            "headword"
        )
        - kbbi_set
    )

    # ==========================================================
    # WHITELIST DASAR
    # ==========================================================

    whitelist_set = set()

    for key in [
        "akronim",
        "daftar_lembaga",
        "daftar_nama_orang",
        "istilah_islam",
    ]:

        col = LEXICON_COL_MAP[key]

        whitelist_set.update(
            to_set(lex_dfs[key], col)
        )

    # ==========================================================
    # DOMAIN VOCAB BERITA UIN 2025
    # ==========================================================

    df_domain = lex_dfs.get(
        "sample_correct_2025",
        pd.DataFrame()
    )

    domain_vocab = set()

    if not df_domain.empty:

        first_col = df_domain.columns[0]

        vals = (
            df_domain[first_col]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # tokenisasi ringan
        for row in vals:

            toks = re.findall(
                r"\b[a-zA-Z][a-zA-Z\-]{2,}\b",
                row
            )

            for tok in toks:

                # skip angka
                if tok.isdigit():
                    continue

                # skip token pendek
                if len(tok) < 3:
                    continue

                domain_vocab.add(tok)

    # ==========================================================
    # MASUKKAN DOMAIN VOCAB KE WHITELIST
    # ==========================================================

    whitelist_set.update(domain_vocab)

    # ==========================================================
    # SERAPAN
    # ==========================================================

    serapan_map = {}
    serapan_set = set()

    df_s = lex_dfs.get(
        "kata_serapan",
        pd.DataFrame()
    )

    if not df_s.empty:

        col_asal = next(
            (
                c for c in df_s.columns
                if "asal" in c.lower()
                or "asing" in c.lower()
            ),
            df_s.columns[0]
        )

        col_serapan = next(
            (
                c for c in df_s.columns
                if "serapan" in c.lower()
                or "hasil" in c.lower()
            ),
            df_s.columns[-1]
        )

        for _, row in df_s.iterrows():

            asal = normalize_token(
                str(row[col_asal])
            )

            serapan = normalize_token(
                str(row[col_serapan])
            )

            if asal and serapan:

                serapan_map[asal] = serapan
                serapan_set.add(asal)

    kbbi_list = sorted(kbbi_set)

    print(f"[DEBUG] domain_vocab: {len(domain_vocab)}")
    print(f"[DEBUG] whitelist total: {len(whitelist_set)}")

    return (
        kbbi_set,
        inggris_set,
        whitelist_set,
        serapan_map,
        serapan_set,
        kbbi_list,
    )

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


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
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
    # [F5] Cek bentuk dasar setelah stripping afiks
    # Justifikasi: Adriani et al. (2007) — Confix-Stripping Stemmer
    if is_kata_berimbuhan_valid(t, kbbi_set):
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
                 serapan_map, serapan_set, kbbi_list,
                 skip_proper_noun: bool = True) -> list:
    sentences = simple_sentence_split(text)
    results   = []

    for sent in sentences:
        tokens = simple_tokenize(sent)
        for pos, tok in enumerate(tokens):
            t = normalize_token(tok)
            if not t or len(t) < 2:
                continue

            # ── Pre-processing filter (reduksi false positive) ──
            skip, skip_reason = should_skip_token(tok, pos, skip_proper_noun)
            if skip:
                results.append({
                    "token"      : tok,
                    "token_norm" : t,
                    "kalimat"    : sent,
                    "flag"       : "SKIPPED",
                    "tipe_error" : skip_reason,
                    "jw_pred"    : "-",
                    "bert_pred"  : "-",
                    "is_error"   : False,
                    "prob_error" : 0.0,
                    "jw_sim"     : 0.0,
                    "rekomendasi": [],
                    "catatan"    : "",
                    "_skipped"   : True,
                })
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

            # Selalu simpan status leksikon untuk highlight, meski not error
            status = jw_res["status"]

            # Tentukan flag visual
            if final_pred == 1:
                if jw_pred == 1 and bert_pred == 0:
                    flag = "TYPO"
                    tipe = "TYPO (leksikal)"
                elif jw_pred == 0 and bert_pred == 1:
                    flag = "REAL_WORD"
                    tipe = "Real-Word Error (kontekstual)"
                else:
                    flag = "TYPO_KONTEKS"
                    tipe = "TYPO + Kontekstual"
            else:
                if status == "KATA_INGGRIS":
                    flag = "KATA_INGGRIS"
                    tipe = "Kata Bahasa Inggris"
                    final_pred = 1  # tandai untuk ditampilkan
                elif status == "KATA_SERAPAN":
                    flag = "KATA_SERAPAN"
                    tipe = "Kata Serapan"
                    final_pred = 1
                elif status == "WHITELIST_KHUSUS":
                    flag = "WHITELIST"
                    tipe = "Nama/Akronim/Istilah Khusus"
                    final_pred = 0  # tidak perlu ditampilkan di tabel error
                else:
                    continue  # kata valid biasa, skip

            recs    = jw_res["top_k_recs"]
            catatan = ""
            if status == "KATA_INGGRIS":
                padanan = serapan_map.get(t)
                catatan = (f"Padanan KBBI: '{padanan}'"
                           if padanan
                           else "Gunakan huruf miring jika dipertahankan")

            results.append({
                "token"      : tok,
                "token_norm" : t,
                "kalimat"    : sent,
                "flag"       : flag,
                "tipe_error" : tipe,
                "jw_pred"    : "ERROR" if jw_pred else "OK",
                "bert_pred"  : "ERROR" if bert_pred else "OK",
                "is_error"   : final_pred == 1 and flag not in ("KATA_INGGRIS", "KATA_SERAPAN"),
                "prob_error" : bert_res["prob_error"],
                "jw_sim"     : jw_res["max_sim"],
                "rekomendasi": recs,
                "catatan"    : catatan,
                "_skipped"   : False,
            })

    return results

# ==============================================================
# HELPER: Render teks dengan highlight + tooltip
# ==============================================================

FLAG_CSS = {
    "TYPO"        : "flag-typo",
    "REAL_WORD"   : "flag-real-word",
    "TYPO_KONTEKS": "flag-typo-konteks",
    "KATA_INGGRIS": "flag-kata-inggris",
    "KATA_SERAPAN": "flag-kata-serapan",
    "WHITELIST"   : "flag-whitelist",
}

FLAG_LABEL = {
    "TYPO"        : "⛔ Typo",
    "REAL_WORD"   : "⚠️ Real-Word Error",
    "TYPO_KONTEKS": "🔴 Typo + Kontekstual",
    "KATA_INGGRIS": "🔵 Kata Inggris",
    "KATA_SERAPAN": "🟢 Kata Serapan",
    "WHITELIST"   : "🟣 Nama/Istilah Khusus",
}

FLAG_COLOR_HEX = {
    "TYPO"        : "#ffe0e0",
    "REAL_WORD"   : "#fff3cd",
    "TYPO_KONTEKS": "#fce7f3",
    "KATA_INGGRIS": "#dbeafe",
    "KATA_SERAPAN": "#d1fae5",
    "WHITELIST"   : "#ede9fe",
}

FLAG_BORDER_HEX = {
    "TYPO"        : "#b91c1c",
    "REAL_WORD"   : "#d97706",
    "TYPO_KONTEKS": "#db2777",
    "KATA_INGGRIS": "#3b82f6",
    "KATA_SERAPAN": "#10b981",
    "WHITELIST"   : "#7c3aed",
}


def build_highlighted_html(text: str, results: list) -> str:
    """
    Bangun HTML teks lengkap dengan kata bermasalah di-highlight.
    Klik kata → panel detail muncul dengan tombol:
      - [Ganti] untuk setiap rekomendasi
      - [Abaikan] untuk menyembunyikan highlight
    Implementasi murni HTML/CSS/JS — tidak memerlukan Streamlit component.
    """
    # Buat index: token_norm -> result dict (hanya yang bukan skipped)
    flagged = {}
    for r in results:
        if r.get("_skipped"):
            continue
        key = r["token_norm"]
        if key not in flagged:
            flagged[key] = r

    # Tokenisasi dengan mempertahankan spasi & tanda baca
    parts = re.split(r"(\s+|[^\w])", text)

    html_parts = []
    panel_parts = []   # panel HTML dikumpulkan terpisah, dirender sekali di bawah

    panel_ids_seen = set()

    for part in parts:
        if not part:
            continue
        part_norm = normalize_token(part)
        if part_norm and part_norm in flagged:
            r      = flagged[part_norm]
            css    = FLAG_CSS.get(r["flag"], "")
            label  = FLAG_LABEL.get(r["flag"], r["flag"])
            recs   = r["rekomendasi"]
            pid    = f"panel_{part_norm}"   # ID panel per kata unik

            # Render token yang bisa diklik
            html_parts.append(
                f'<span class="word-wrap">'
                f'<span class="token-badge {css}" '
                f'onclick="openPanel(\'{pid}\')" '
                f'title="Klik untuk detail">'
                f'{part}</span>'
                f'</span>'
            )

            # Buat panel sekali per kata unik
            if pid not in panel_ids_seen:
                panel_ids_seen.add(pid)

                # Baris rekomendasi sebagai tombol klik
                rec_html = ""
                if recs:
                    rec_html = "<div style='margin-top:6px'><b>📋 Saran perbaikan:</b><br>"
                    for rec in recs[:5]:
                        # onclick: ganti semua kemunculan kata ini di teks asli
                        rec_html += (
                            f'<span class="rec-btn" '
                            f'onclick="replaceToken(\'{part_norm}\',\'{rec}\',\'{pid}\')">'
                            f'{rec}</span>'
                        )
                    rec_html += "</div>"
                else:
                    rec_html = "<div style='margin-top:6px;color:#6c7086'>Tidak ada saran spesifik.</div>"

                catatan_html = ""
                if r["catatan"]:
                    catatan_html = f"<div style='margin-top:4px;color:#f9e2af'>ℹ️ {r['catatan']}</div>"

                panel_parts.append(f"""
<div id="{pid}" class="token-panel">
  <span class="panel-close" onclick="closePanel('{pid}')">✕</span>
  <b>{label}</b><br>
  <span style="color:#89b4fa">Token:</span> <code style="color:#cba6f7">{part_norm}</code>
  {catatan_html}
  <hr class="panel-divider">
  <small style="color:#6c7086">JW sim: {r['jw_sim']} &nbsp;|&nbsp; BERT prob: {r['prob_error']:.4f}</small>
  {rec_html}
  <div>
    <span class="ignore-btn" onclick="ignoreToken('{part_norm}','{pid}')">✓ Abaikan</span>
  </div>
</div>
""")
        elif re.match(r"\s+", part):
            html_parts.append(part)
        else:
            html_parts.append(f'<span class="token-normal">{part}</span>')

    # Gabungkan: overlay + teks + semua panel + JS
    overlay_html = '<div id="panel-overlay" class="panel-overlay" onclick="closeAllPanels()"></div>'

    js = """
<script>
function openPanel(pid) {
    closeAllPanels();
    var p = document.getElementById(pid);
    if (p) { p.classList.add('active'); }
    document.getElementById('panel-overlay').classList.add('active');
}
function closePanel(pid) {
    var p = document.getElementById(pid);
    if (p) { p.classList.remove('active'); }
    document.getElementById('panel-overlay').classList.remove('active');
}
function closeAllPanels() {
    document.querySelectorAll('.token-panel').forEach(function(p){
        p.classList.remove('active');
    });
    document.getElementById('panel-overlay').classList.remove('active');
}
function replaceToken(original, replacement, pid) {
    // Tandai semua span dengan token ini sebagai "sudah diganti"
    document.querySelectorAll('.token-badge').forEach(function(el) {
        if (el.textContent.trim().toLowerCase() === original) {
            el.textContent = replacement;
            el.className = 'token-badge';   // hapus warna flag
            el.style.background = '#d1fae5';
            el.style.color = '#065f46';
            el.style.borderBottom = '2px solid #10b981';
            el.title = 'Diganti: ' + replacement;
            el.onclick = null;
        }
    });
    closePanel(pid);
}
function ignoreToken(original, pid) {
    document.querySelectorAll('.token-badge').forEach(function(el) {
        if (el.textContent.trim().toLowerCase() === original) {
            el.className = 'token-ignored';
            el.onclick = null;
        }
    });
    closePanel(pid);
}
</script>
"""

    panels_html = "\n".join(panel_parts)
    text_html   = "".join(html_parts)

    return overlay_html + text_html + panels_html + js


def render_legend(flags_present: set):
    """Render legenda warna yang muncul sesuai flag yang ada di hasil."""
    items = ""
    for flag, label in FLAG_LABEL.items():
        if flag in flags_present:
            color = FLAG_COLOR_HEX[flag]
            border = FLAG_BORDER_HEX[flag]
            items += (
                f'<div class="legend-item">'
                f'<div class="legend-dot" style="background:{color};'
                f'border:2px solid {border};"></div>'
                f'{label}</div>'
            )
    if items:
        st.markdown(
            f'<div class="legend-grid">{items}</div>',
            unsafe_allow_html=True,
        )

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

    st.markdown("**Tampilan Hasil**")
    show_inggris  = st.toggle("Tampilkan kata Inggris",  value=True)
    show_serapan  = st.toggle("Tampilkan kata serapan",  value=True)
    show_tabel    = st.toggle("Tampilkan tabel detail",  value=True)

    st.markdown("---")

    st.markdown("**Filter Pre-processing**")
    st.caption(
        "Filter ini diterapkan sebelum token masuk ke model, "
        "untuk mengurangi false positive pada entitas yang bukan "
        "target deteksi typo."
    )
    skip_proper_noun = st.toggle(
        "Lewati nama orang/tempat (huruf kapital)",
        value=True,
        help=(
            "Token berawalan huruf kapital di tengah kalimat "
            "dianggap Named Entity (nama orang, tempat, institusi) "
            "dan dilewati. Referensi: Jurafsky & Martin (2023), Ch. 8."
        ),
    )
    st.caption(
        "🔒 Filter gelar akademik (S.Pd., M.A., dll) dan "
        "akronim ALL-CAPS selalu aktif secara otomatis."
    )

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
            "Contoh: Rektor UIN Jakarta menyambut positif pencapaian ini. "
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
    file_text = ""
    run_file  = False
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
            skip_proper_noun=skip_proper_noun,
        )
        elapsed = round(time.time() - t0, 2)

    # Filter: pisahkan skipped dan yang perlu ditampilkan
    n_skipped = sum(1 for r in results if r.get("_skipped"))
    results_display = []
    for r in results:
        if r.get("_skipped"):
            continue
        if r["flag"] == "KATA_INGGRIS" and not show_inggris:
            continue
        if r["flag"] == "KATA_SERAPAN" and not show_serapan:
            continue
        results_display.append(r)

    st.markdown("---")

    # ── Metrik ringkasan ─────────────────────────────────────
    total_tok = len([t for t in simple_tokenize(
        re.sub(r"[^\w\s]", " ", text_to_run)) if len(t) >= 2])
    n_err     = sum(1 for r in results_display
                    if r["flag"] in ("TYPO", "REAL_WORD", "TYPO_KONTEKS"))
    n_flag    = len(results_display)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Token", total_tok)
    c2.metric("Token Bermasalah", n_err)
    c3.metric("Token Diflag", n_flag)
    c4.metric("Token Di-skip", n_skipped,
              help="Token yang dilewati filter pre-processing (gelar, akronim, nama proper)")
    c5.metric("Waktu Analisis", f"{elapsed}s")

    # ── Teks hasil highlight ──────────────────────────────────
    st.markdown("### 📄 Teks dengan Anotasi")
    st.markdown(
        "_Arahkan kursor ke kata yang ditandai untuk melihat detail dan saran perbaikan._",
        unsafe_allow_html=False,
    )

    flags_present = {r["flag"] for r in results_display}
    render_legend(flags_present)

    if results_display:
        highlighted_html = build_highlighted_html(text_to_run, results_display)
        st.markdown(
            f'<div class="text-preview-box">{highlighted_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ Tidak ditemukan kesalahan atau token yang perlu ditandai.", icon="✅")

    # ── Tabel detail ──────────────────────────────────────────
    if show_tabel and results_display:
        st.markdown("### 📊 Tabel Hasil Deteksi")

        # Hanya tampilkan error (bukan kata inggris/serapan) di tabel
        results_err = [r for r in results_display
                       if r["flag"] in ("TYPO", "REAL_WORD", "TYPO_KONTEKS")]

        if results_err:
            tabel = pd.DataFrame([{
                "Token"          : r["token"],
                "Flag"           : FLAG_LABEL.get(r["flag"], r["flag"]),
                "Tipe Error"     : r["tipe_error"],
                "JW"             : r["jw_pred"],
                "BERT"           : r["bert_pred"],
                "Skor JW"        : r["jw_sim"],
                "Prob Error BERT": r["prob_error"],
                "Rekomendasi"    : ", ".join(r["rekomendasi"]) if r["rekomendasi"] else "-",
                "Catatan"        : r["catatan"] or "-",
            } for r in results_err])

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
        else:
            st.info("Tidak ada error leksikal/kontekstual yang terdeteksi. "
                    "Token yang diflag adalah kata Inggris atau kata serapan.")

        # ── Detail per token (accordion) ─────────────────────
        if results_err:
            st.markdown("### 🔎 Detail Per Token")
            for r in results_err:
                icon = {
                    "TYPO"        : "🔴",
                    "REAL_WORD"   : "🟡",
                    "TYPO_KONTEKS": "🟠",
                }.get(r["flag"], "⚪")
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
