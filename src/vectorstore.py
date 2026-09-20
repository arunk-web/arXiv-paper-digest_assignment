"""
Minimal local vector store on top of FAISS + sentence-transformers.
One index per paper, persisted to disk under DATA_DIR so the QA session
can be resumed without re-embedding.
"""

import os
import json

import pickle
from typing import List, Tuple

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim) 
        self.meta: List[dict] = []          

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9

        return vecs / norms

    def add(self, texts: List[str], metas: List[dict]):
        embedder = get_embedder()

        vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = self._normalize(vecs.astype("float32"))
        self.index.add(vecs)

        self.meta.extend(metas)

    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        embedder = get_embedder()
        qvec = embedder.encode([query], convert_to_numpy=True).astype("float32")
        qvec = self._normalize(qvec)

        scores, idxs = self.index.search(qvec, min(k, len(self.meta)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):

            if idx == -1:
                continue
            results.append((self.meta[idx], float(score)))
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta.pkl", "wb") as f:

            pickle.dump(self.meta, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        index = faiss.read_index(path + ".faiss")
        with open(path + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)

        vs = cls(dim=index.d)
        vs.index = index
        vs.meta = meta
        
        return vs
