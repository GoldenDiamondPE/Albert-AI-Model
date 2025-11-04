import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import threading

# TEST

_MODEL_PATH = os.path.join("models", "albert-bi-encoder")

_model = None
_model_lock = threading.Lock()

def _load_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    _model = SentenceTransformer(_MODEL_PATH, device=device)
                except Exception:
                    # Fallback if fine-tuned model not found yet
                    _model = SentenceTransformer("albert-base-v2", device=device)
    return _model

def _extract_text(r: dict) -> str:
    # Be defensive about different result shapes
    title = r.get("title") or r.get("heading") or ""
    snippet = r.get("snippet") or r.get("description") or r.get("text") or ""
    return (title + " " + snippet).strip()

def rank_results(query: str, results: list, batch_size: int = 32):
    if not results:
        return results

    model = _load_model()

    docs = [_extract_text(r) for r in results]
    # If everything is empty, keep original order
    if not any(docs):
        return results

    # Encode (normalized so cosine == dot product)
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    d_vecs = model.encode(docs, normalize_embeddings=True, batch_size=batch_size)

    scores = np.dot(d_vecs, q_vec)  # shape [num_docs]
    ranked = sorted(zip(scores.tolist(), results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked]