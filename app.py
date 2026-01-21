import os
import re
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pymupdf  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from mistralai import Mistral


st.set_page_config(page_title="Lecture Saver (NN)", layout="wide")


DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def normalize_text_keep_paragraphs(text: str) -> str:
    """Light cleanup but keeps paragraph structure for better chunking."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # join hyphenated words across line breaks: "back-\nprop" -> "backprop"
    text = re.sub(r"-\s*\n\s*(\w)", r"\1", text)

    # remove repeated spaces inside lines
    text = re.sub(r"[ \t]{2,}", " ", text)

    # normalize too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_units(text: str) -> List[str]:
    text = normalize_text_keep_paragraphs(text)
    if not text:
        return []

    # Make bullets their own boundaries
    text = re.sub(r"[•▪◦●■◆]\s*", "\n", text)

    # First try paragraph split
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    # If paragraphs are not meaningful, sentence-split (Arabic + English punctuation)
    if len(blocks) <= 1:
        flat = re.sub(r"\n+", " ", text).strip()
        blocks = [b.strip() for b in re.split(r"(?<=[\.\!\?\u061F\u061B])\s+", flat) if b.strip()]

    units: List[str] = []
    for b in blocks:
        # If a block is huge, split further by punctuation
        if len(b) > 1200:
            parts = [p.strip() for p in re.split(r"(?<=[\.\!\?\u061F\u061B])\s+", b) if p.strip()]
            units.extend(parts if parts else [b])
        else:
            units.append(b)

    # remove tiny noise units
    units = [u for u in units if len(u) >= 20]
    return units


def semantic_chunk_text(
    text: str,
    model: SentenceTransformer,
    max_chars: int = 1800,
    min_chars: int = 350,
    similarity_threshold: float = 0.55,
    overlap_units: int = 1,
) -> List[str]:
    units = split_into_units(text)
    if not units:
        return []

    if len(units) == 1:
        return [units[0]]

    emb = model.encode(units, normalize_embeddings=True, show_progress_bar=False)
    sims = (emb[:-1] * emb[1:]).sum(axis=1)  # cosine since normalized

    chunks: List[str] = []
    cur_units = [units[0]]
    cur_len = len(units[0])

    for i in range(1, len(units)):
        u = units[i]
        ulen = len(u) + 1

        boundary = False
        if cur_len + ulen > max_chars:
            boundary = True
        else:
            if cur_len >= min_chars and float(sims[i - 1]) < similarity_threshold:
                boundary = True

        if boundary:
            chunk = "\n".join(cur_units).strip()
            if chunk:
                chunks.append(chunk)

            # overlap
            if overlap_units > 0:
                cur_units = cur_units[-overlap_units:]
                cur_len = sum(len(x) + 1 for x in cur_units)
            else:
                cur_units = []
                cur_len = 0

        cur_units.append(u)
        cur_len += ulen

    last = "\n".join(cur_units).strip()
    if last:
        chunks.append(last)

    # merge very small trailing chunks
    merged: List[str] = []
    for ch in chunks:
        if merged and len(ch) < int(min_chars * 0.6):
            merged[-1] = (merged[-1] + "\n" + ch).strip()
        else:
            merged.append(ch)

    return merged


# ---------------------------
# PDF loading (with optional OCR)
# ---------------------------
def extract_page_text(page, use_ocr: bool, ocr_lang: str, min_chars_for_no_ocr: int) -> Tuple[str, bool, str]:
    """
    Returns: (text, used_ocr, ocr_error_message)
    """
    used_ocr = False
    ocr_error = ""

    # Try normal extraction
    try:
        text = page.get_text("text")
    except Exception:
        text = page.get_text()

    text_norm = normalize_text_keep_paragraphs(text)
    if (not use_ocr) or (len(text_norm) >= min_chars_for_no_ocr):
        return text_norm, used_ocr, ocr_error

    # If text is too small, try OCR (if supported)
    try:
        if hasattr(page, "get_textpage_ocr"):
            tp = page.get_textpage_ocr(language=ocr_lang)
            text2 = page.get_text("text", textpage=tp)
            used_ocr = True
            return normalize_text_keep_paragraphs(text2), used_ocr, ocr_error
        else:
            ocr_error = "PyMuPDF OCR غير مدعوم في نسختك (لا يوجد get_textpage_ocr)."
            return text_norm, used_ocr, ocr_error
    except Exception as e:
        ocr_error = f"OCR failed: {e}"
        return text_norm, used_ocr, ocr_error


def load_pdf_pages(
    pdf_path: str,
    use_ocr: bool = False,
    ocr_lang: str = "ara+eng",
    min_chars_for_no_ocr: int = 60,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = pymupdf.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    try:
        for i, page in enumerate(doc):
            text, used_ocr, ocr_err = extract_page_text(page, use_ocr, ocr_lang, min_chars_for_no_ocr)
            pages.append(
                {
                    "page": i + 1,
                    "source": os.path.basename(pdf_path),
                    "text": text,
                }
            )
            report.append(
                {
                    "source": os.path.basename(pdf_path),
                    "page": i + 1,
                    "chars": len(text),
                    "used_ocr": used_ocr,
                    "ocr_error": ocr_err,
                }
            )
    finally:
        doc.close()

    return pages, report


def semantic_chunk_pages(
    pages: List[Dict[str, Any]],
    model: SentenceTransformer,
    max_chars: int,
    min_chars: int,
    similarity_threshold: float,
    overlap_units: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for p in pages:
        page_chunks = semantic_chunk_text(
            p["text"],
            model=model,
            max_chars=max_chars,
            min_chars=min_chars,
            similarity_threshold=similarity_threshold,
            overlap_units=overlap_units,
        )
        for idx, ch in enumerate(page_chunks):
            chunks.append(
                {
                    "text": ch,
                    "source": p["source"],
                    "page": p["page"],
                    "chunk_id": f"{p['source']}:p{p['page']}:c{idx}",
                }
            )
    return chunks

def index_chunks_in_chroma(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    persist_dir: str,
    batch_size: int = 128,
) -> int:
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection("lectures")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        "lectures",
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []
    for ch in chunks:
        if ch["text"].strip():
            ids.append(ch["chunk_id"])
            docs.append(ch["text"])
            metas.append({"source": ch["source"], "page": ch["page"]})

    for i in range(0, len(docs), batch_size):
        bd = docs[i : i + batch_size]
        bi = ids[i : i + batch_size]
        bm = metas[i : i + batch_size]

        be = model.encode(bd, normalize_embeddings=True, show_progress_bar=False).tolist()
        collection.add(ids=bi, documents=bd, metadatas=bm, embeddings=be)

    return collection.count()


def retrieve(
    question: str,
    model: SentenceTransformer,
    persist_dir: str,
    k: int = 10,
) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("lectures")

    q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "distance": float(dist),
            }
        )
    return out

def answer_with_mistral(question: str, retrieved: List[Dict[str, Any]]) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "⚠️ لم يتم العثور على مفتاح Mistral. ضع MISTRAL_API_KEY في Environment."

    client = Mistral(api_key=api_key)

    if not retrieved:
        return "لا أعرف. (لم يتم استرجاع أي مقاطع من المحاضرات)"

    context_parts = []
    sources = set()
    for r in retrieved:
        context_parts.append(f"[Source: {r['source']} , Page: {r['page']}]\n{r['text']}\n")
        sources.add((r["source"], r["page"]))

    context = "\n\n".join(context_parts)
    sources_lines = [f"{src} (Page {pg})" for (src, pg) in sorted(sources)]
    sources_text = "\n".join(f"{i+1}. {line}" for i, line in enumerate(sources_lines))

    messages = [
        {
            "role": "system",
            "content": (
                "أنت مدرس مساعد. أجب فقط باستخدام النص الموجود داخل Context. "
                "إذا لم تجد الجواب صراحة في Context أعد الجملة التالية فقط بدون أي إضافات: لا أعرف. "
                "إذا أجبت، ضع سطر (اقتباس داعم) من الـ Context يثبت الجواب."
            ),
        },
        {"role": "user", "content": f"السؤال: {question}\n\nContext:\n{context}"},
    ]

    resp = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()

    return answer + "\n\nSources:\n" + sources_text


# ---------------------------
# UI
# ---------------------------
st.title("📚 Lecture-Saver 3000")

st.sidebar.header("Indexing")

pdf_paths = st.sidebar.text_area(
    "PDF paths (one per line)",
    value=(
        "lectures/NN-Theoretical-Lec 1+2.pdf\n"
        "lectures/NN-Theoretical-Lec3.pdf\n"
        "lectures/NN-Theoretical-Lec4.pdf\n"
        "lectures/NN-Theoretical-Lec5.pdf\n"
        "lectures/NN-Theoretical-Lec6 .pdf\n"
        "lectures/NN-Theoretical-Lec7.pdf\n"
        "lectures/NN-Theoretical-Lec8.pdf\n"
        "lectures/NN-Theoretical-Lec9.pdf"
    ),
).splitlines()
pdf_paths = [p.strip() for p in pdf_paths if p.strip()]

st.sidebar.subheader("Embedding model")
embed_model_name = st.sidebar.selectbox(
    "Choose model",
    options=[
        DEFAULT_EMBED_MODEL,
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
    index=0,
)

st.sidebar.subheader("OCR (اختياري)")
use_ocr = st.sidebar.checkbox("Enable OCR for low-text pages", value=False)
ocr_lang = st.sidebar.text_input("OCR languages", value="ara+eng")
min_chars_for_no_ocr = st.sidebar.slider("If page text chars < this, try OCR", 10, 300, 60, 10)

st.sidebar.subheader("Semantic chunk settings")
max_chars = st.sidebar.slider("Max chunk size (chars)", 600, 3500, 1800, 100)
min_chars = st.sidebar.slider("Min chunk size before semantic split", 100, 1500, 350, 50)
sim_th = st.sidebar.slider("Similarity threshold (lower = more splits)", 0.30, 0.90, 0.55, 0.01)
overlap_units = st.sidebar.slider("Overlap units", 0, 4, 1, 1)
batch_size = st.sidebar.slider("Index embed batch size", 32, 512, 128, 32)

st.sidebar.subheader("Retrieval")
k = st.sidebar.slider("Top-k retrieved chunks", 2, 25, 10, 1)
show_debug = st.sidebar.checkbox("Show retrieval debug", value=True)
show_index_report = st.sidebar.checkbox("Show indexing report (chars/pages)", value=True)

persist_dir = "chroma_db"


if st.sidebar.button("Build / Rebuild Index"):
    model = get_embedding_model(embed_model_name)

    all_pages: List[Dict[str, Any]] = []
    all_report: List[Dict[str, Any]] = []

    with st.spinner("Loading PDFs..."):
        for p in pdf_paths:
            if not os.path.exists(p):
                st.sidebar.error(f"File not found: {p}")
                continue
            pages, report = load_pdf_pages(
                p,
                use_ocr=use_ocr,
                ocr_lang=ocr_lang,
                min_chars_for_no_ocr=min_chars_for_no_ocr,
            )
            all_pages.extend(pages)
            all_report.extend(report)

    if show_index_report and all_report:
        weak = [r for r in all_report if r["chars"] < 60]
        st.sidebar.write(f"Loaded pages: {len(all_report)}")
        st.sidebar.write(f"Low-text pages (<60 chars): {len(weak)}")
        if weak:
            st.sidebar.warning("صفحات نصها قليل جدًا (قد تكون صور وتحتاج OCR):")
            for r in weak[:30]:
                st.sidebar.write(f"- {r['source']} p{r['page']} chars={r['chars']} ocr={r['used_ocr']}")

    with st.spinner("Semantic chunking..."):
        chunks = semantic_chunk_pages(
            all_pages,
            model=model,
            max_chars=max_chars,
            min_chars=min_chars,
            similarity_threshold=sim_th,
            overlap_units=overlap_units,
        )

    with st.spinner("Indexing in Chroma..."):
        n = index_chunks_in_chroma(
            chunks,
            model=model,
            persist_dir=persist_dir,
            batch_size=batch_size,
        )

    st.sidebar.success(f"Indexed {n} chunks using: {embed_model_name}")


question = st.text_input("Ask a question about your lectures:")

if st.button("Ask") and question.strip():
    model = get_embedding_model(embed_model_name)

    with st.spinner("Retrieving relevant lecture notes..."):
        retrieved = retrieve(question, model=model, persist_dir=persist_dir, k=k)

    if show_debug:
        st.markdown("### Retrieved chunks (debug)")
        for r in retrieved:
            st.write(
                {
                    "source": r["source"],
                    "page": r["page"],
                    "distance": r["distance"],  # smaller is better
                    "preview": (r["text"][:250] + ("..." if len(r["text"]) > 250 else "")),
                }
            )
        st.markdown("---")

    with st.spinner("Asking Mistral..."):
        response = answer_with_mistral(question, retrieved)

    st.markdown("### Answer")
    st.write(response)
