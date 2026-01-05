# src/rag.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from pypdf import PdfReader

from openai import OpenAI


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]  # e.g. {"source": "...", "page": 3}


@dataclass
class RAGIndex:
    index: faiss.Index
    chunks: List[Chunk]
    dim: int
    embed_model: str


# ---------------------------
# Loaders
# ---------------------------

def load_text_from_pdf(pdf_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns a list of (page_text, meta) where meta includes page number (1-indexed).
    """
    reader = PdfReader(pdf_path)
    pages: List[Tuple[str, Dict[str, Any]]] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        meta = {"source": os.path.basename(pdf_path), "page": i + 1}
        pages.append((text, meta))

    return pages


def load_text_from_txt(txt_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [(text, {"source": os.path.basename(txt_path), "page": None})]


def load_guide(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Load the tourism guide and return a list of (text, meta).
    For PDFs: returns one entry per page.
    For TXTs: returns one entry total.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_text_from_pdf(path)
    if ext in (".txt", ".md"):
        return load_text_from_txt(path)
    raise ValueError(f"Unsupported guide format: {ext}. Use .pdf or .txt/.md")


# ---------------------------
# Chunking
# ---------------------------

def _clean_text(s: str) -> str:
    # Minimal cleanup to reduce weird spacing
    return " ".join((s or "").replace("\t", " ").split())


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Good enough for a first functional version.
    """
    text = _clean_text(text)
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap

    return chunks


def make_chunks(
    pages: List[Tuple[str, Dict[str, Any]]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Convert (page_text, meta) to Chunk objects with stable ids.
    """
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    out: List[Chunk] = []
    for page_idx, (page_text, meta) in enumerate(pages):
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for j, ch in enumerate(page_chunks):
            # id encodes page + chunk number for citation
            cid = f"p{meta.get('page') or (page_idx+1)}_c{j:03d}"
            out.append(Chunk(id=cid, text=ch, meta=dict(meta)))

    return out


# ---------------------------
# Embeddings
# ---------------------------

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")
    return OpenAI(api_key=api_key)


def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Returns embeddings as float32 numpy array of shape (n, dim).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    client = _get_openai_client()
    vectors: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        # Keep original order:
        batch_vecs = [d.embedding for d in resp.data]
        vectors.extend(batch_vecs)

    arr = np.array(vectors, dtype=np.float32)
    return arr


# ---------------------------
# FAISS index
# ---------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Basic cosine similarity via L2 on normalized vectors.
    We'll use IndexFlatIP and normalize embeddings.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a 2D non-empty array")

    # Normalize for cosine similarity with inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_rag_index(
    guide_path: str,
    embed_model: str = "text-embedding-3-small",
) -> RAGIndex:
    pages = load_guide(guide_path)
    chunks = make_chunks(pages)

    texts = [c.text for c in chunks]
    embs = embed_texts(texts, model=embed_model)

    index = build_faiss_index(embs)
    return RAGIndex(index=index, chunks=chunks, dim=embs.shape[1], embed_model=embed_model)


def save_rag_index(rag: RAGIndex, dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    faiss.write_index(rag.index, os.path.join(dir_path, "faiss.index"))

    meta = {
        "dim": rag.dim,
        "embed_model": rag.embed_model,
        "chunks": [
            {"id": c.id, "text": c.text, "meta": c.meta}
            for c in rag.chunks
        ],
    }
    with open(os.path.join(dir_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_rag_index(dir_path: str) -> RAGIndex:
    index_path = os.path.join(dir_path, "faiss.index")
    chunks_path = os.path.join(dir_path, "chunks.json")

    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = [Chunk(id=x["id"], text=x["text"], meta=x["meta"]) for x in meta["chunks"]]
    return RAGIndex(index=index, chunks=chunks, dim=int(meta["dim"]), embed_model=str(meta["embed_model"]))


# ---------------------------
# Retrieval
# ---------------------------

def retrieve(
    rag: RAGIndex,
    query: str,
    top_k: Optional[int] = None,
    embed_model: Optional[str] = None,
) -> List[Chunk]:
    if top_k is None:
        top_k = int(os.getenv("TOP_K", "5"))
    if embed_model is None:
        embed_model = embed_model or rag.embed_model

    q_emb = embed_texts([query], model=embed_model)
    faiss.normalize_L2(q_emb)

    scores, idxs = rag.index.search(q_emb, top_k)
    idxs = idxs[0].tolist()

    results: List[Chunk] = []
    for idx in idxs:
        if idx == -1:
            continue
        results.append(rag.chunks[idx])

    return results


def format_context(chunks: List[Chunk], max_chars: int = 4000) -> str:
    """
    Builds a context string you can inject into the LLM prompt.
    Includes citations like [p3_c002 | pág. 3].
    """
    parts: List[str] = []
    used = 0

    for c in chunks:
        page = c.meta.get("page")
        cite = f"[{c.id}" + (f" | pág. {page}]" if page else "]")
        snippet = f"{cite}\n{c.text}\n"
        if used + len(snippet) > max_chars:
            break
        parts.append(snippet)
        used += len(snippet)

    return "\n".join(parts).strip()


'''
COMO USARLO EN EL NOTEBOOK

from src.rag import build_rag_index, save_rag_index, load_rag_index, retrieve, format_context

GUIDE_PATH = "data/guia_turistica.pdf"   # o .txt
INDEX_DIR = "data/index_faiss"

# Construir (una vez)
rag = build_rag_index(GUIDE_PATH)
save_rag_index(rag, INDEX_DIR)

# Cargar (en ejecuciones posteriores)
rag = load_rag_index(INDEX_DIR)

# Recuperar contexto
chunks = retrieve(rag, "¿Qué puedo visitar en el centro histórico?", top_k=5)
context = format_context(chunks)
print(context)

'''