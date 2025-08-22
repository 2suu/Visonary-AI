# services/text_embedder.py
from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache()
def get_text_embedder():
    return SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")

def encode_texts(texts: list[str]):
    model = get_text_embedder()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
