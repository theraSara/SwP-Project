import argparse
import os
import re
import pandas as pd
from wordfreq import zipf_frequency


def normalize_word(w):
    if pd.isna(w):
        return None
    w = str(w).strip().lower()
    if not w:
        return None
    return w


def is_good_word_filtered(w, min_len=2, max_len=15, min_zipf=3.0):
    if not w:
        return False

    w = w.lower().strip()

    if not re.fullmatch(r"[a-z]+", w):
        return False

    if not (min_len <= len(w) <= max_len):
        return False

    if len(w) >= 5 and sum(ch in "aeiouy" for ch in w) == 0:
        return False

    if re.search(r"(.)\1\1", w):
        return False

    if zipf_frequency(w, "en") < min_zipf:
        return False

    return True


def load_nltk_words():
    try:
        from nltk.corpus import words
    except Exception as e:
        raise RuntimeError(
            "Could not import nltk.corpus.words. "
            "Install nltk and run nltk.download('words')."
        ) from e

    return list(words.words())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bk21_csv", type=str, required=True,
                        help="Path to BK21 stimuli CSV with critical_word column")
    parser.add_argument("--output_txt", type=str, required=True,
                        help="Output lexicon txt file (one word per line)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional output csv with metadata")

    parser.add_argument("--use_nltk_words", action="store_true",
                        help="Include filtered NLTK words")
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--allow_apostrophe", action="store_true")
    parser.add_argument("--min_zipf", type=float, default=3.0)

    args = parser.parse_args()

    bk21 = pd.read_csv(args.bk21_csv)

    if "critical_word" not in bk21.columns:
        raise ValueError("BK21 CSV must contain a 'critical_word' column.")

    lexicon_rows = []
    seen = set()

    # 1) Always include BK21 critical words
    bk21_words = bk21["critical_word"].dropna().astype(str).tolist()

    for w in bk21_words:
        w_norm = normalize_word(w)
        if w_norm is None:
            continue

        if w_norm not in seen:
            seen.add(w_norm)
            lexicon_rows.append({
                "word": w_norm,
                "source": "bk21",
                "passes_filter": True
            })

    # 2) Optionally include filtered NLTK words
    if args.use_nltk_words:
        nltk_words = load_nltk_words()

        for w in nltk_words:
            w_norm = normalize_word(w)
            if w_norm is None:
                continue

            if not is_good_word_filtered(
                w_norm,
                min_len=args.min_len,
                max_len=args.max_len,
                min_zipf=args.min_zipf
            ):
                continue

            if w_norm not in seen:
                seen.add(w_norm)
                lexicon_rows.append({
                    "word": w_norm,
                    "source": "nltk_words",
                    "passes_filter": True
                })

    lexicon_df = pd.DataFrame(lexicon_rows).sort_values("word").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for w in lexicon_df["word"]:
            f.write(w + "\n")

    print(f"Saved lexicon txt to: {args.output_txt}")
    print(f"Total words: {len(lexicon_df)}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        lexicon_df.to_csv(args.output_csv, index=False)
        print(f"Saved lexicon metadata csv to: {args.output_csv}")


if __name__ == "__main__":
    main()