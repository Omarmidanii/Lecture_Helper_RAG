# 📚 Lecture Saver 3000 (RAG for PDF Lectures)

A Streamlit app that lets you ask questions about your lecture PDFs using **RAG (Retrieval-Augmented Generation)**:

- Extracts text from PDF lectures
- Splits content into **semantic chunks**
- Stores chunks in **ChromaDB** with **SentenceTransformers embeddings**
- Retrieves the most relevant chunks for your question
- Uses **Mistral** to answer **based only on retrieved context**

---

## ✨ Features

- ✅ Index multiple PDFs (one per line)
- ✅ Semantic chunking (better than fixed character slicing)
- ✅ Multilingual embeddings (works better with Arabic/English)
- ✅ ChromaDB persistent storage
- ✅ Source citations (PDF + page)
- ✅ Debug mode: view retrieved chunks & distances
- ✅ Optional OCR for low-text / scanned PDF pages (Arabic + English)

---

## 🧠 How it works (Pipeline)

1. **Load PDFs** using PyMuPDF
2. **Extract text** (optional OCR if text is too small)
3. **Semantic chunking** by paragraph/sentence units + embedding similarity boundaries
4. **Embeddings** via `sentence-transformers` (multilingual recommended)
5. **Store** chunks + metadata in ChromaDB
6. **Retrieve** top-k chunks for the question
7. **Generate answer** with Mistral using retrieved context only

---

## 📦 Tech Stack

- **UI:** Streamlit
- **PDF parsing:** PyMuPDF (`pymupdf`)
- **Vector DB:** ChromaDB
- **Embeddings:** SentenceTransformers
- **LLM:** Mistral (`mistral-large-latest`)

---

## ✅ Requirements

- Python 3.10+ recommended
- A **Mistral API Key**

Optional (for OCR):

- PyMuPDF OCR support + Tesseract installed

---

## ⚙️ Installation

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```
