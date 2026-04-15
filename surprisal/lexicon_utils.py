import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def normalize_word(w):
    if pd.isna(w):
        return None
    w = str(w).strip().lower()
    if not w:
        return None
    return w


def load_words_from_file(path):
    words = []
    seen = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = normalize_word(line)
            if not w:
                continue
            if w not in seen:
                seen.add(w)
                words.append(w)

    return words


def load_nltk_words_basic():
    try:
        import nltk
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words")
        from nltk.corpus import words
    except Exception as e:
        raise RuntimeError(
            "Could not load nltk words corpus. Install nltk and run nltk.download('words')."
        ) from e

    out = []
    seen = set()

    for w in words.words():
        w = normalize_word(w)
        if not w:
            continue
        if re.fullmatch(r"[a-z]+", w):
            if w not in seen:
                seen.add(w)
                out.append(w)

    return out


def load_wordfreq_topn(n=100000):
    try:
        from wordfreq import top_n_list
    except Exception as e:
        raise RuntimeError(
            "Could not import wordfreq.top_n_list. Please install wordfreq."
        ) from e

    raw_words = top_n_list("en", n)

    out = []
    seen = set()

    for w in raw_words:
        w = normalize_word(w)
        if not w:
            continue
        if re.fullmatch(r"[a-z]+", w):
            if w not in seen:
                seen.add(w)
                out.append(w)

    return out


def extract_dataset_targets(df, target_col):
    out = []
    seen = set()

    for x in df[target_col].dropna().astype(str).tolist():
        w = normalize_word(x)
        if not w:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)

    return out


def merge_words_preserve_order(*word_lists):
    out = []
    seen = set()

    for words in word_lists:
        for w in words:
            w = normalize_word(w)
            if not w:
                continue
            if w not in seen:
                seen.add(w)
                out.append(w)

    return out


def build_candidate_words(mode, df, target_col, lexicon_file=None, wordfreq_topn=100000):
    dataset_targets = extract_dataset_targets(df, target_col)

    if mode == "filtered_nltk":
        if not lexicon_file:
            raise ValueError("lexicon_file is required for mode='filtered_nltk'")
        base_words = load_words_from_file(lexicon_file)

    elif mode == "raw_nltk":
        base_words = load_nltk_words_basic()

    elif mode == "wordfreq":
        base_words = load_wordfreq_topn(wordfreq_topn)

    else:
        raise ValueError(f"Unknown lexicon mode: {mode}")

    return merge_words_preserve_order(base_words, dataset_targets)


def build_candidate_groups(tokenizer, candidate_words, device):
    groups = defaultdict(lambda: {"words": [], "ids": []})
    candidate_word_set = set()

    for w in tqdm(candidate_words, desc="Tokenizing candidate lexicon", leave=False):
        ids = tokenizer(" " + w, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] == 0:
            continue

        tok_len = ids.shape[1]
        groups[tok_len]["words"].append(w)
        groups[tok_len]["ids"].append(ids)
        candidate_word_set.add(w)

    return groups, candidate_word_set