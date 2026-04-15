import gc
import os
import re
import math
import argparse
import traceback
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from surprisal_utils import safe_model_name, safe_mode_name, normalize_word, cleanup_model


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

    from wordfreq import zipf_frequency
    if zipf_frequency(w, "en") < min_zipf:
        return False

    return True

def load_filtered_nltk_dynamic(min_len=2, max_len=15, min_zipf=3.0):
    raw_words = load_nltk_words_basic()
    out = []
    seen = set()

    for w in raw_words:
        if is_good_word_filtered(w, min_len=min_len, max_len=max_len, min_zipf=min_zipf):
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


def build_candidate_words(args, df, mode):
    dataset_targets = extract_dataset_targets(df, args.target_col)

    if mode == "filtered_nltk":
        if not args.lexicon_file:
            raise ValueError("--lexicon_file is required for mode=filtered_nltk")
        base_words = load_words_from_file(args.lexicon_file)

    elif mode == "raw_nltk":
        base_words = load_nltk_words_basic()

    elif mode == "wordfreq":
        base_words = load_wordfreq_topn(args.wordfreq_topn)

    else:
        raise ValueError(f"Unknown lexicon mode: {mode}")

    candidate_words = merge_words_preserve_order(base_words, dataset_targets)
    return candidate_words

def build_candidate_groups(tokenizer, candidate_words, device):
    """
    Pre-tokenize candidate words and group them by token length.
    """
    groups = defaultdict(lambda: {"words": [], "ids": []})
    candidate_word_set = set()

    for w in tqdm(candidate_words, desc="Tokenizing candidate lexicon", leave=False):
        w = str(w).strip()
        if not w:
            continue

        ids = tokenizer(" " + w, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] == 0:
            continue

        tok_len = ids.shape[1]
        groups[tok_len]["words"].append(w)
        groups[tok_len]["ids"].append(ids)
        candidate_word_set.add(w)

    return groups, candidate_word_set


def load_model_and_tokenizer(model_id, force_no_quant=False):
    SMALL_MODEL_THRESHOLD = 1_000_000_000

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        dummy = AutoModelForCausalLM.from_config(config)
    n_params = sum(p.numel() for p in dummy.parameters())
    del dummy

    use_quantization = n_params >= SMALL_MODEL_THRESHOLD and not force_no_quant

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        print(f"Loading model {model_id} ({n_params/1e9:.1f}B params) in 8-bit mode...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif n_params >= SMALL_MODEL_THRESHOLD:
        print(f"Loading model {model_id} ({n_params/1e9:.1f}B params) in bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print(f"Loading model {model_id} ({n_params/1e6:.0f}M params) in fp32...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )

    device = next(model.parameters()).device
    model.eval()
    print(f"Model loaded on device: {device}")

    return model, tokenizer, device

def compute_uni_surprisal_word(model, tokenizer, prefix, target, device):
    """
    Word-level unidirectional surprisal:
    -log2 P(target_word | prefix)
    computed as sum over target tokens.
    """
    if pd.isna(prefix) or pd.isna(target):
        return None

    prefix_str = str(prefix)
    target_str = " " + str(target).strip()

    prefix_ids = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    target_ids = tokenizer(target_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    full_ids = torch.cat([prefix_ids, target_ids], dim=1)
    p_len = prefix_ids.shape[1]
    f_len = full_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits[0]

    total_uni = 0.0
    token_data = []

    for pos in range(p_len, f_len):
        token_id = full_ids[0, pos].item()
        step_logits = logits[pos - 1].float()
        log_probs = F.log_softmax(step_logits, dim=-1)
        probs = F.softmax(step_logits, dim=-1)

        token_surprisal = -log_probs[token_id].item() / math.log(2)
        total_uni += token_surprisal

        entropy_bits = -torch.sum(probs * (log_probs / math.log(2))).item()

        token_data.append({
            "token_id": token_id,
            "token_str": tokenizer.decode([token_id]),
            "surprisal": token_surprisal,
            "entropy": entropy_bits,
        })

    return {
        "uni_val": total_uni,
        "token_data": token_data,
    }


def score_candidate_batch(model, prefix_ids, cand_ids_batch, suffix_ids):
    """
    Score log P(candidate_tokens + suffix | prefix) for a batch of candidates
    that all have the same token length.

    Returns:
        [B] tensor of log-probabilities in float64
    """
    B = cand_ids_batch.shape[0]
    p_len = prefix_ids.shape[1]
    cand_len = cand_ids_batch.shape[1]
    s_len = suffix_ids.shape[1]

    batch_prefix = prefix_ids.expand(B, -1)
    batch_suffix = suffix_ids.expand(B, -1)

    full_ids = torch.cat([batch_prefix, cand_ids_batch, batch_suffix], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits

    total_lp = torch.zeros(B, device=logits.device, dtype=torch.float64)

    for j in range(cand_len):
        step_logits = logits[:, p_len - 1 + j, :].float()
        step_log_probs = F.log_softmax(step_logits, dim=-1).double()
        token_ids = cand_ids_batch[:, j].unsqueeze(1)
        total_lp += step_log_probs.gather(1, token_ids).squeeze(1)

    for j in range(s_len):
        step_logits = logits[:, p_len + cand_len - 1 + j, :].float()
        step_log_probs = F.log_softmax(step_logits, dim=-1).double()
        token_ids = batch_suffix[:, j].unsqueeze(1)
        total_lp += step_log_probs.gather(1, token_ids).squeeze(1)

    return total_lp


def compute_bi_surprisal_word_batched(
    model,
    tokenizer,
    prefix,
    target,
    suffix,
    candidate_groups,
    candidate_word_set,
    device,
    cand_batch_size=128,
):
    """
    Word-level cloze surprisal under a causal LM:

        P(target | prefix, suffix)
        =
        P(target, suffix | prefix)
        /
        sum_w P(w, suffix | prefix)

    surprisal = -log2 P(target | prefix, suffix)
    """
    if pd.isna(prefix) or pd.isna(target):
        return float("nan")

    prefix_str = str(prefix).rstrip()
    target_word = str(target).strip()
    suffix_text = "" if pd.isna(suffix) else str(suffix).strip()
    suffix_str = (" " + suffix_text) if suffix_text else ""

    prefix_ids = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    suffix_ids = tokenizer(suffix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    target_in_lexicon = target_word in candidate_word_set
    log_num = None
    den_terms = []

    for tok_len in sorted(candidate_groups.keys()):
        group = candidate_groups[tok_len]
        ids_list = group["ids"]
        words_list = group["words"]
        n = len(ids_list)

        for start in range(0, n, cand_batch_size):
            end = min(start + cand_batch_size, n)
            batch_ids = ids_list[start:end]
            batch_words = words_list[start:end]

            cand_ids_batch = torch.cat(batch_ids, dim=0)
            batch_lp = score_candidate_batch(model, prefix_ids, cand_ids_batch, suffix_ids)

            den_terms.append(batch_lp.detach().cpu())

            if target_in_lexicon and log_num is None:
                for local_idx, w in enumerate(batch_words):
                    if w == target_word:
                        log_num = batch_lp[local_idx].item()
                        break

    if not target_in_lexicon:
        target_ids = tokenizer(
            " " + target_word,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(device)

        target_lp = score_candidate_batch(model, prefix_ids, target_ids, suffix_ids)
        log_num = target_lp[0].item()
        den_terms.append(target_lp.detach().cpu())

    if log_num is None:
        print(f"[ERROR] Could not find target '{target_word}' in denominator candidates.")
        return float("nan")

    all_den = torch.cat(den_terms).double()
    log_den = torch.logsumexp(all_den, dim=0).item()

    bi_val = -(log_num - log_den) / math.log(2)

    if bi_val < 0:
        if abs(bi_val) < 1e-9:
            bi_val = 0.0
        else:
            print(
                f"[ERROR] Unexpected negative BI surprisal: {bi_val:.10f} "
                f"(log_num={log_num:.6f}, log_den={log_den:.6f})"
            )
            bi_val = 0.0

    return bi_val


def run_for_model_and_mode(model_id, lexicon_mode, args):
    print("=" * 90)
    print(f"RUNNING MODEL: {model_id}")
    print(f"LEXICON MODE : {lexicon_mode}")
    print("=" * 90)

    df = pd.read_csv(args.input_csv)

    model, tokenizer, device = load_model_and_tokenizer(
        model_id,
        force_no_quant=args.no_quantization
    )

    candidate_words = build_candidate_words(args, df, lexicon_mode)
    print(f"Loaded {len(candidate_words)} candidate words for mode={lexicon_mode}")

    candidate_groups, candidate_word_set = build_candidate_groups(
        tokenizer, candidate_words, device
    )

    print("Built candidate groups by token length:")
    for tok_len in sorted(candidate_groups.keys()):
        print(f"  token_len={tok_len}: {len(candidate_groups[tok_len]['words'])} words")

    model_name = safe_model_name(model_id)
    mode_name = safe_mode_name(lexicon_mode)

    output_csv = os.path.join(
        args.output_dir,
        f"{model_name}__{mode_name}__wordlevel.csv"
    )
    checkpoint_path = output_csv.replace(".csv", "_checkpoint.csv")
    metadata_path = output_csv.replace(".csv", "_metadata.pt")

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)

        uni_ok = checkpoint_df[args.output_col_uni].notna()
        bi_ok = checkpoint_df[args.output_col_bi].notna()

        completed_indices = set(checkpoint_df.index[uni_ok & bi_ok])

        df[args.output_col_uni] = checkpoint_df[args.output_col_uni]
        df[args.output_col_bi] = checkpoint_df[args.output_col_bi]

        print(f"Resuming from checkpoint: {len(completed_indices)}/{len(df)} rows done.")
    else:
        completed_indices = set()
        df[args.output_col_uni] = float("nan")
        df[args.output_col_bi] = float("nan")

    if os.path.exists(metadata_path):
        all_meta = torch.load(metadata_path, weights_only=False)
        metadata_list = [m for m in all_meta if m["index"] in completed_indices]
    else:
        metadata_list = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name} | {mode_name}"):
        if i in completed_indices:
            continue

        try:
            res = compute_uni_surprisal_word(
                model=model,
                tokenizer=tokenizer,
                prefix=row[args.prefix_col],
                target=row[args.target_col],
                device=device,
            )
            uni_val = res["uni_val"] if res else float("nan")

            bi_val = compute_bi_surprisal_word_batched(
                model=model,
                tokenizer=tokenizer,
                prefix=row[args.prefix_col],
                target=row[args.target_col],
                suffix=row[args.suffix_col],
                candidate_groups=candidate_groups,
                candidate_word_set=candidate_word_set,
                device=device,
                cand_batch_size=args.cand_batch_size,
            )

            df.at[i, args.output_col_uni] = uni_val
            df.at[i, args.output_col_bi] = bi_val

            metadata_list.append({
                "index": i,
                "model_id": model_id,
                "lexicon_mode": lexicon_mode,
                "uni_surprisal_word": uni_val,
                "bi_surprisal_word": bi_val,
                "token_data": res["token_data"] if res else None,
                "cand_batch_size": args.cand_batch_size,
                "lexicon_size": len(candidate_word_set),
                "wordfreq_topn": args.wordfreq_topn if lexicon_mode == "wordfreq" else None,
            })

            if i % args.save_every == 0:
                df.to_csv(checkpoint_path, index=False)
                torch.save(metadata_list, metadata_path)

        except KeyboardInterrupt:
            print(f"\nInterrupted at row {i}. Saving checkpoint...")
            df.to_csv(checkpoint_path, index=False)
            torch.save(metadata_list, metadata_path)
            print(f"Checkpoint saved to {checkpoint_path}. Re-run to resume.")
            cleanup_model(model, tokenizer)
            return

        except Exception as e:
            print(f"Row {i} failed: {e}")
            traceback.print_exc()
            df.at[i, args.output_col_uni] = float("nan")
            df.at[i, args.output_col_bi] = float("nan")

    df.to_csv(output_csv, index=False)
    torch.save(metadata_list, metadata_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"Done! CSV saved to {output_csv}")
    print(f"Metadata saved to {metadata_path}")

    cleanup_model(model, tokenizer)
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_ids", type=str, nargs="+", required=True)

    parser.add_argument(
        "--lexicon_modes",
        type=str,
        nargs="+",
        required=True,
        choices=["filtered_nltk", "raw_nltk", "wordfreq"],
    )

    parser.add_argument(
        "--lexicon_file",
        type=str,
        default=None,
        help="Required for lexicon_mode=filtered_nltk"
    )

    parser.add_argument("--output_col_uni", type=str, default="uni_surprisal_word")
    parser.add_argument("--output_col_bi", type=str, default="bi_surprisal_word")

    parser.add_argument("--prefix_col", type=str, default="prefix")
    parser.add_argument("--target_col", type=str, default="target_llm")
    parser.add_argument("--suffix_col", type=str, default="suffix")

    parser.add_argument("--cand_batch_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument("--wordfreq_topn", type=int, default=100000,
                        help="Number of top English words to use for lexicon_mode=wordfreq")

    parser.add_argument("--no_quantization", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_id in args.model_ids:
        for lexicon_mode in args.lexicon_modes:
            run_for_model_and_mode(
                model_id=model_id,
                lexicon_mode=lexicon_mode,
                args=args,
            )


if __name__ == "__main__":
    main()