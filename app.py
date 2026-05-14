from __future__ import annotations

import html as html_lib
import os
import re
import tempfile
import time
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from config import DRIVE_IDS, JW_CFG, MODEL_CFG

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
# STYLE VISUAL
# ==============================================================

FLAG_STYLES = {
    "TYPO": {
        "label": "Salah ketik",
        "bg": "#ffd6d6",
        "border": "#d64545",
        "text": "#7a1111",
    },
    "REAL_WORD": {
        "label": "Salah konteks",
        "bg": "#eadcff",
        "border": "#8b5cf6",
        "text": "#4c1d95",
    },
    "KATA_INGGRIS": {
        "label": "Kata asing",
        "bg": "#fff2b3",
        "border": "#d4a017",
        "text": "#6b4f00",
    },
    "KATA_SERAPAN": {
        "label": "Kata serapan",
        "bg": "#dbeafe",
        "border": "#3b82f6",
        "text": "#1e3a8a",
    },
    "WHITELIST_KHUSUS": {
        "label": "Khusus/whitelist",
        "bg": "#dcfce7",
        "border": "#16a34a",
        "text": "#14532d",
    },
    "TIDAK_DIKENAL": {
        "label": "Tidak dikenal",
        "bg": "#fecaca",
        "border": "#ef4444",
        "text": "#7f1d1d",
    },
}

MODEL_LABELS = {
    0: "OK",
    1: "ERROR",
}

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


def extract_sentence_spans(text: str) -> List[Tuple[str, int, int]]:
    """Kembalikan daftar (kalimat, start, end) dengan posisi di teks asli."""
    spans: List[Tuple[str, int, int]] = []
    if not text or not text.strip():
        return spans

    pattern = re.compile(r".+?(?:[.!?](?:\s+|$)|$)", flags=re.DOTALL)
    for match in pattern.finditer(text):
        sent = match.group().strip()
        if sent:
            spans.append((sent, match.start(), match.end()))
    return spans


def tokenize_with_spans(sentence: str) -> List[Tuple[str, int, int]]:
    """Token kata pada sebuah kalimat, lengkap dengan posisi relatif."""
    return [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\b\w+\b", sentence, flags=re.UNICODE)
    ]


def escape_multiline(text: str) -> str:
    return html_lib.escape(text).replace("\n", "<br>")


# ==============================================================
# PATH LOKAL UNTUK CACHE UNDUHAN
# ==============================================================

_TMP = tempfile.gettempdir()
MODEL_LOCAL = os.path.join(_TMP, "model_indobert_best")
LEXICON_LOCAL = os.path.join(_TMP, "leksikon")
os.makedirs(MODEL_LOCAL, exist_ok=True)
os.makedirs(LEXICON_LOCAL, exist_ok=True)

LEXICON_COL_MAP = {
    "kbbi": "kata",
    "kata_inggris": "headword",
    "akronim": "akronim",
    "daftar_lembaga": "Nama Lembaga",
    "daftar_nama_orang": "Nama",
    "istilah_islam": "Kata",
}

# ==============================================================
# LOAD RESOURCES
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_LOCAL)
    model = BertForSequenceClassification.from_pretrained(
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

    lex_dfs = {}
    for key in [
        "kbbi",
        "kata_inggris",
        "kata_serapan",
        "akronim",
        "daftar_lembaga",
        "daftar_nama_orang",
        "istilah_islam",
    ]:
        local_path = os.path.join(LEXICON_LOCAL, f"{key}.csv")
        if not os.path.exists(local_path):
            gdown.download(id=DRIVE_IDS[key], output=local_path, quiet=True)
        lex_dfs[key] = read_csv_safe(local_path)

    kbbi_set = to_set(lex_dfs["kbbi"], "kata")
    inggris_set = to_set(lex_dfs["kata_inggris"], "headword") - kbbi_set

    whitelist_set = set()
    for key in ["akronim", "daftar_lembaga", "daftar_nama_orang", "istilah_islam"]:
        col = LEXICON_COL_MAP[key]
        whitelist_set.update(to_set(lex_dfs[key], col))

    serapan_map: Dict[str, str] = {}
    serapan_set = set()
    df_s = lex_dfs.get("kata_serapan", pd.DataFrame())
    if not df_s.empty:
        col_asal = next(
            (c for c in df_s.columns if "asal" in c.lower() or "asing" in c.lower()),
            df_s.columns[0],
        )
        col_serapan = next(
            (c for c in df_s.columns if "serapan" in c.lower() or "hasil" in c.lower()),
            df_s.columns[-1],
        )
        for _, row in df_s.iterrows():
            asal = normalize_token(str(row[col_asal]))
            serapan = normalize_token(str(row[col_serapan]))
            if asal and serapan:
                serapan_map[asal] = serapan
                serapan_set.add(asal)

    kbbi_list = sorted(kbbi_set)
    return kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list


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
    matches = 0
    transpositions = 0

    for i in range(len1):
        for j in range(max(0, i - match_dist), min(i + match_dist + 1, len2)):
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

    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    jaro = jaro_similarity(s1, s2)
    prefix = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


# ==============================================================
# KLASIFIKASI TOKEN
# ==============================================================


def classify_token(token: str, kbbi_set, inggris_set, whitelist_set, serapan_set) -> str:
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


# ==============================================================
# PREDIKSI JARO-WINKLER
# ==============================================================


def predict_jw(token: str, kbbi_set, inggris_set, whitelist_set, serapan_set, kbbi_list, threshold: float, top_k: int = 5) -> dict:
    t = normalize_token(token)
    status = classify_token(t, kbbi_set, inggris_set, whitelist_set, serapan_set)

    if status in ("WHITELIST_KHUSUS", "KBBI_VALID", "KATA_SERAPAN", "KOSONG"):
        return {
            "pred": 0,
            "max_sim": 1.0,
            "best_match": t,
            "top_k_recs": [],
            "status": status,
        }

    sims = [(w, jaro_winkler_similarity(t, w)) for w in kbbi_list]
    sims.sort(key=lambda x: x[1], reverse=True)
    best_word, best_sim = sims[0]
    pred = 0 if best_sim >= threshold else 1

    return {
        "pred": pred,
        "max_sim": round(best_sim, 4),
        "best_match": best_word,
        "top_k_recs": [w for w, _ in sims[:top_k]],
        "status": status,
    }


# ==============================================================
# PREDIKSI INDOBERT
# ==============================================================


def predict_bert(kalimat: str, token: str, tokenizer, model, device) -> dict:
    text_a = clean_whitespace(kalimat)
    text_b = normalize_token(token)
    enc = tokenizer(
        text_a,
        text_b,
        max_length=MODEL_CFG["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            token_type_ids=enc["token_type_ids"].to(device),
        )
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return {
        "pred": pred,
        "prob_correct": round(float(probs[0]), 4),
        "prob_error": round(float(probs[1]), 4),
    }


# ==============================================================
# KEPUTUSAN HYBRID & PENANDA TAMPILAN
# ==============================================================


def decide_final_pred(model_choice: str, jw_pred: int, bert_pred: int) -> int:
    if model_choice == "Jaro-Winkler":
        return jw_pred
    if model_choice == "IndoBERT":
        return bert_pred
    return 1 if (jw_pred == 1 or bert_pred == 1) else 0


def resolve_visual_flag(jw_status: str, jw_pred: int, bert_pred: int, final_pred: int) -> Optional[str]:
    """Label visual yang ditampilkan di teks. Tidak mengubah hasil eksperimen."""
    if jw_status == "WHITELIST_KHUSUS":
        return "WHITELIST_KHUSUS"
    if jw_status == "KATA_SERAPAN":
        return "KATA_SERAPAN"
    if jw_status == "KATA_INGGRIS":
        return "KATA_INGGRIS"
    if final_pred == 1:
        if jw_pred == 1 and bert_pred == 0:
            return "TYPO"
        if jw_pred == 0 and bert_pred == 1:
            return "REAL_WORD"
        return "TYPO"
    return None


# ==============================================================
# PIPELINE ANALISIS TEKS
# ==============================================================


def analyze_text(
    text: str,
    model_choice: str,
    tokenizer,
    bert_model,
    device,
    kbbi_set,
    inggris_set,
    whitelist_set,
    serapan_map,
    serapan_set,
    kbbi_list,
) -> Tuple[List[dict], List[dict]]:
    """Return: (semua token yang dianalisis, token yang ditandai)."""
    sentence_spans = extract_sentence_spans(text)
    all_rows: List[dict] = []
    flagged_rows: List[dict] = []

    for sent, sent_start, sent_end in sentence_spans:
        tokens = tokenize_with_spans(sent)
        for tok, start, end in tokens:
            t = normalize_token(tok)
            if not t or len(t) < 2:
                continue
            if t.isdigit():
                continue

            jw_res = predict_jw(
                t,
                kbbi_set,
                inggris_set,
                whitelist_set,
                serapan_set,
                kbbi_list,
                threshold=JW_CFG["threshold"],
                top_k=JW_CFG["top_k"],
            )
            bert_res = predict_bert(sent, t, tokenizer, bert_model, device)

            jw_pred = jw_res["pred"]
            bert_pred = bert_res["pred"]
            final_pred = decide_final_pred(model_choice, jw_pred, bert_pred)
            visual_flag = resolve_visual_flag(jw_res["status"], jw_pred, bert_pred, final_pred)

            # Keterangan ringkas untuk tooltip
            alasan = []
            if jw_res["status"] == "KATA_INGGRIS":
                padanan = serapan_map.get(t)
                if padanan:
                    alasan.append(f"Padanan KBBI: {padanan}")
                else:
                    alasan.append("Gunakan huruf miring jika dipertahankan")
            elif jw_res["status"] == "KATA_SERAPAN":
                alasan.append("Terdeteksi sebagai kata serapan")
            elif jw_res["status"] == "WHITELIST_KHUSUS":
                alasan.append("Termasuk whitelist/kata khusus")
            elif final_pred == 1 and jw_pred == 1 and bert_pred == 0:
                alasan.append("Diduga typo/leksikal")
            elif final_pred == 1 and jw_pred == 0 and bert_pred == 1:
                alasan.append("Diduga real-word error/kontekstual")
            elif final_pred == 1:
                alasan.append("Diduga error gabungan")

            recs = jw_res["top_k_recs"]
            if recs:
                alasan.append("Rekomendasi: " + ", ".join(recs[:3]))

            row = {
                "token": tok,
                "token_norm": t,
                "kalimat": sent,
                "sent_start": sent_start,
                "start": sent_start + start,
                "end": sent_start + end,
                "jw_status": jw_res["status"],
                "visual_flag": visual_flag,
                "tipe_error": (
                    "TYPO (leksikal)"
                    if jw_pred == 1 and bert_pred == 0
                    else "Real-Word Error (kontekstual)"
                    if jw_pred == 0 and bert_pred == 1
                    else "TYPO + Kontekstual"
                    if final_pred == 1
                    else "-"
                ),
                "jw_pred": MODEL_LABELS[jw_pred],
                "bert_pred": MODEL_LABELS[bert_pred],
                "final_pred": MODEL_LABELS[final_pred],
                "prob_error": bert_res["prob_error"],
                "prob_correct": bert_res["prob_correct"],
                "jw_sim": jw_res["max_sim"],
                "best_match": jw_res["best_match"],
                "rekomendasi": recs,
                "catatan": " | ".join(alasan),
                "highlight": visual_flag is not None,
            }
            all_rows.append(row)
            if row["highlight"]:
                flagged_rows.append(row)

    return all_rows, flagged_rows


# ==============================================================
# RENDER TEKS BERWARNA
# ==============================================================


def build_tooltip(row: dict) -> str:
    lines = [
        f"Token: {row['token']}",
        f"Label: {FLAG_STYLES.get(row['visual_flag'], {}).get('label', row['visual_flag'] or '-')}",
        f"JW: {row['jw_pred']} (sim={row['jw_sim']})",
        f"BERT: {row['bert_pred']} (prob error={row['prob_error']})",
        f"Final: {row['final_pred']}",
    ]
    if row.get("catatan"):
        lines.append(row["catatan"])
    return "\n".join(lines)


def render_highlighted_text(text: str, rows: List[dict], show_non_error: bool = True) -> str:
    by_start = {}
    for r in rows:
        if not show_non_error and r["visual_flag"] in {"KATA_INGGRIS", "KATA_SERAPAN", "WHITELIST_KHUSUS"}:
            continue
        by_start[r["start"]] = r

    parts: List[str] = []
    cursor = 0
    for m in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        parts.append(escape_multiline(text[cursor:m.start()]))
        row = by_start.get(m.start())
        token_html = html_lib.escape(m.group())
        if row and row.get("highlight"):
            style = FLAG_STYLES.get(row["visual_flag"], FLAG_STYLES["TIDAK_DIKENAL"])
            tooltip = escape_multiline(build_tooltip(row))
            span = (
                f'<span title="{tooltip}" '
                f'style="background:{style["bg"]}; color:{style["text"]}; '
                f'border:1px solid {style["border"]}; border-radius:6px; '
                f'padding:1px 5px; font-weight:600; white-space:nowrap;">'
                f"{token_html}</span>"
            )
            parts.append(span)
        else:
            parts.append(token_html)
        cursor = m.end()
    parts.append(escape_multiline(text[cursor:]))

    return (
        '<div style="line-height:1.95; font-size:1.02rem; white-space:pre-wrap; ' 
        'word-break:break-word;">'
        + "".join(parts)
        + "</div>"
    )


def render_legend() -> None:
    chips = []
    for key in ["TYPO", "REAL_WORD", "KATA_INGGRIS", "KATA_SERAPAN", "WHITELIST_KHUSUS"]:
        s = FLAG_STYLES[key]
        chips.append(
            f'<span style="display:inline-block; margin:0 10px 10px 0; padding:4px 10px; '
            f'border-radius:999px; background:{s["bg"]}; color:{s["text"]}; '
            f'border:1px solid {s["border"]}; font-size:0.92rem;">'
            f'{s["label"]}</span>'
        )
    st.markdown(
        "<div style='margin-top:4px; margin-bottom:8px;'><b>Legenda warna:</b> "
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


# ==============================================================
# ANTARMUKA STREAMLIT
# ==============================================================

with st.sidebar:
    st.markdown("### 📝 Sistem Penyuntingan Kata")
    st.caption("Berita UIN Jakarta · Demo Streamlit")
    st.markdown("---")

    model_choice = st.selectbox(
        "Pilih Model Deteksi",
        options=["Hybrid-OR", "IndoBERT", "Jaro-Winkler"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Performa Model (test set)**")
    perf = {
        "Hybrid-OR": {"F1": "0.9976", "Recall": "0.9988", "Precision": "0.9964"},
        "IndoBERT": {"F1": "0.9952", "Recall": "0.9940", "Precision": "0.9964"},
        "Jaro-Winkler": {"F1": "0.9824", "Recall": "0.9654", "Precision": "1.0000"},
    }
    for metric, val in perf[model_choice].items():
        st.metric(metric, val)

    st.markdown("---")
    show_non_error = st.checkbox(
        "Tampilkan juga kata asing/serapan/whitelist",
        value=True,
        help="Jika aktif, kata asing, kata serapan, dan kata khusus juga diberi warna agar mudah dibaca.",
    )

    st.markdown("---")
    st.caption("Noeni Indah Sulistiyani\nTeknik Informatika · UIN Jakarta")

st.title("📝 Sistem Rekomendasi Penyuntingan Kata")
st.markdown(
    "Deteksi dan penandaan kata pada teks berita universitas menggunakan **Jaro-Winkler** dan **IndoBERT**."
)
st.markdown("---")

with st.spinner("Memuat model dan leksikon..."):
    tokenizer, bert_model, device = load_model()
    kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list = load_lexicons()

st.success(f"Model **{model_choice}** siap digunakan.", icon="✅")

# Tabs
input_tab, file_tab = st.tabs(["✏️ Input Teks", "📂 Upload File"])

with input_tab:
    input_text = st.text_area(
        "Masukkan teks berita:",
        height=200,
        placeholder=(
            "Contoh: Rektor UIN Jakarta menyambut positif pencpaaian ini. "
            "Menurutnya capaian ini merupakan bagian dari upaya berkelanjutan "
            "universitas dalam memperkuat kualitas academic di tingkat global."
        ),
    )
    run_text = st.button("🔍 Analisis", type="primary", use_container_width=True, key="btn_text")

with file_tab:
    uploaded = st.file_uploader("Upload file berita (.txt atau .docx)", type=["txt", "docx"])
    file_text = ""
    run_file = False
    if uploaded:
        if uploaded.name.endswith(".txt"):
            file_text = uploaded.read().decode("utf-8", errors="replace")
        else:
            import docx as _docx

            doc = _docx.Document(uploaded)
            file_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        st.text_area("Isi file:", value=file_text, height=200, disabled=True)
        run_file = st.button("🔍 Analisis File", type="primary", use_container_width=True, key="btn_file")

text_to_run = ""
if run_text and input_text.strip():
    text_to_run = input_text
elif run_file and file_text.strip():
    text_to_run = file_text

if text_to_run:
    with st.spinner(f"Menganalisis dengan {model_choice}..."):
        t0 = time.time()
        all_rows, flagged_rows = analyze_text(
            text_to_run,
            model_choice,
            tokenizer,
            bert_model,
            device,
            kbbi_set,
            inggris_set,
            whitelist_set,
            serapan_map,
            serapan_set,
            kbbi_list,
        )
        elapsed = round(time.time() - t0, 2)

    st.markdown("---")

    total_words = len([m.group() for m in re.finditer(r"\b\w+\b", text_to_run, flags=re.UNICODE) if len(m.group()) >= 2])
    n_flagged = len(flagged_rows)
    n_error = len([r for r in flagged_rows if r["visual_flag"] in {"TYPO", "REAL_WORD", "TIDAK_DIKENAL"}])
    n_foreign = len([r for r in flagged_rows if r["visual_flag"] == "KATA_INGGRIS"])
    n_serapan = len([r for r in flagged_rows if r["visual_flag"] == "KATA_SERAPAN"])
    n_whitelist = len([r for r in flagged_rows if r["visual_flag"] == "WHITELIST_KHUSUS"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Token", total_words)
    c2.metric("Token Ditandai", n_flagged)
    c3.metric("Error Final", n_error)
    c4.metric("Waktu Analisis", f"{elapsed}s")

    render_legend()

    st.markdown("### Teks Hasil Deteksi")
    st.markdown(render_highlighted_text(text_to_run, all_rows, show_non_error=show_non_error), unsafe_allow_html=True)

    if n_flagged == 0:
        st.success("✅ Tidak ditemukan kata yang perlu ditandai.", icon="✅")
    else:
        st.info(
            f"Ringkasan: {n_error} error final, {n_foreign} kata asing, {n_serapan} kata serapan, {n_whitelist} kata khusus.",
            icon="ℹ️",
        )

        st.markdown("### Tabel Hasil Deteksi")
        df_flagged = pd.DataFrame(flagged_rows)
        if not df_flagged.empty:
            tabel = df_flagged[
                [
                    "token",
                    "visual_flag",
                    "tipe_error",
                    "jw_pred",
                    "bert_pred",
                    "final_pred",
                    "jw_sim",
                    "prob_error",
                    "best_match",
                    "rekomendasi",
                    "catatan",
                ]
            ].copy()
            tabel["rekomendasi"] = tabel["rekomendasi"].apply(lambda x: ", ".join(x) if x else "-")
            tabel.rename(
                columns={
                    "token": "Token",
                    "visual_flag": "Flag",
                    "tipe_error": "Tipe Error",
                    "jw_pred": "JW",
                    "bert_pred": "BERT",
                    "final_pred": "Final",
                    "jw_sim": "Skor JW",
                    "prob_error": "Prob Error BERT",
                    "best_match": "Kandidat Terdekat",
                    "rekomendasi": "Top-k Rekomendasi",
                    "catatan": "Catatan",
                },
                inplace=True,
            )
            st.dataframe(
                tabel,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Skor JW": st.column_config.NumberColumn(format="%.4f"),
                    "Prob Error BERT": st.column_config.NumberColumn(format="%.4f", min_value=0, max_value=1),
                },
            )

        st.markdown("### Detail Per Token")
        for r in flagged_rows:
            flag_key = r["visual_flag"] or "TIDAK_DIKENAL"
            icon = "🔴" if flag_key in {"TYPO", "REAL_WORD", "TIDAK_DIKENAL"} else "🟡"
            label = FLAG_STYLES.get(flag_key, FLAG_STYLES["TIDAK_DIKENAL"])["label"]
            with st.expander(f"{icon} **{r['token']}** — {label}"):
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown(f"**Token:** `{r['token']}`")
                    st.markdown(f"**Flag:** {flag_key}")
                    st.markdown(f"**Tipe:** {r['tipe_error']}")
                    st.markdown(f"**JW:** {r['jw_pred']} (sim = {r['jw_sim']})")
                    st.markdown(f"**IndoBERT:** {r['bert_pred']} (prob error = {r['prob_error']})")
                    st.markdown(f"**Final:** {r['final_pred']}")
                    if r["catatan"]:
                        st.info(r["catatan"])

                with col_r:
                    st.markdown("**Rekomendasi kata (top-k):**")
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
                    count=1,
                )
                st.markdown(f"> {highlighted}")
