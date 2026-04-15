import gc
import pandas as pd

import torch

def safe_model_name(model_id):
    return model_id.replace("/", "_")


def safe_mode_name(mode):
    return mode.replace("/", "_")


def normalize_word(w):
    if pd.isna(w):
        return None
    w = str(w).strip().lower()
    if not w:
        return None
    return w


def cleanup_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()