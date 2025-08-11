import os
import numpy as np
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, models
from config import SENTENCE_MODEL_PATH, LINEAR_LAYER_PATH, DEVICE

_base_model = None
_linear = None

def _is_sbert_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))

def get_models():
    global _base_model, _linear
    if _base_model is None:
        path = SENTENCE_MODEL_PATH
        if os.path.isdir(path) and not os.path.exists(os.path.join(path, "modules.json")):
            # 순수 HF 포맷 → Transformer + Pooling 구성
            word = models.Transformer(path)
            pooling = models.Pooling(
                word.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
            )
            _base_model = SentenceTransformer(modules=[word, pooling], device=DEVICE)
        else:
            _base_model = SentenceTransformer(path, device=DEVICE)

        _linear = nn.Linear(768, 103)
        _linear.load_state_dict(torch.load(LINEAR_LAYER_PATH, map_location=DEVICE))
        _linear.to(DEVICE).eval()
    return _base_model, _linear
