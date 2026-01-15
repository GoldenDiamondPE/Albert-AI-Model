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
    #checks if there are results to rank
    if not results:
        return results
    # Load model
    model = _load_model()
    # format results
    docs = [_extract_text(r) for r in results]
    # check for empty query or docs
    if not any(docs) or not query:
        return results
    else:
        # Encode (normalized so cosine == dot product)
        #q_vec = model.encode(query, normalize_embeddings=True)
        # list of scores for all results
        scores = []
        for result in range(len(docs)):
            # final score for a result
            final_score = 0
            # goes through each query text
            for text in range(len(query)):
                h_vec = model.encode(query[text], normalize_embeddings=True)
                d_vecs = model.encode(docs[result], normalize_embeddings=True)
                score = np.dot(d_vecs, h_vec)  # shape [num_doc]
                # add to final score
                final_score += score
            # appends document's final score to scores list
            scores.append(final_score)
        # rank results by score
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        # return only results, in ranked order
        return [r for _, r in ranked]
