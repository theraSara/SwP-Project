import argparse
import math
import os
import traceback
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm


def score_continuation_logprob(model, prefix_ids, continuation_ids):
    """
    Returns log P(continuation_ids | prefix_ids) in natural log.
    prefix_ids: [1, p_len]
    continuation_ids: [1, c_len]
    """
    full_ids = torch.cat([prefix_ids, continuation_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # [seq_len, vocab]

    p_len = prefix_ids.shape[1]
    c_len = continuation_ids.shape[1]

    total_lp = 0.0
    for pos in range(c_len):
        token_id = continuation_ids[0, pos].item()
        step_logits = logits[p_len - 1 + pos].float()
        log_probs = F.log_softmax(step_logits, dim=-1)
        total_lp += log_probs[token_id].item()

    return total_lp


def compute_uni_surprisal_word(model, tokenizer, prefix, target, device):
    """
    Whole-word unidirectional surprisal:
    -log2 P(target_word_tokens | prefix)
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


def load_candidate_words(lexicon_file):
    words = []
    seen = set()

    with open(lexicon_file, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            if w not in seen:
                seen.add(w)
                words.append(w)

    return words


def build_candidate_cache(tokenizer, candidate_words, device):
    """
    Pre-tokenize candidate words once.
    Returns list of tuples: (word, token_ids_tensor [1, len])
    """
    cache = []
    seen = set()

    for w in tqdm(candidate_words, desc="Tokenizing candidate lexicon"):
        w = str(w).strip()
        if not w or w in seen:
            continue
        seen.add(w)

        ids = tokenizer(" " + w, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # skip candidates that tokenize to empty
        if ids.shape[1] == 0:
            continue

        cache.append((w, ids))

    return cache


def compute_bi_surprisal_word(
    model,
    tokenizer,
    prefix,
    target,
    suffix,
    candidate_cache,
    candidate_word_set,
    device,
):
    """
    Whole-word bidirectional surprisal:

    P(target_word | prefix, suffix)
    = P(target_word_tokens, suffix | prefix)
      / sum_{w in candidate_words} P(tokens(w), suffix | prefix)

    Returns surprisal in bits.
    """
    if pd.isna(prefix) or pd.isna(target) or pd.isna(suffix):
        return float("nan")

    prefix_str = str(prefix).rstrip()
    target_word = str(target).strip()
    suffix_text = str(suffix).strip()

    suffix_str = (" " + suffix_text) if suffix_text else ""

    prefix_ids = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    suffix_ids = tokenizer(suffix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # numerator: target word + suffix
    target_ids = tokenizer(" " + target_word, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    true_continuation = torch.cat([target_ids, suffix_ids], dim=1)

    log_num = score_continuation_logprob(model, prefix_ids, true_continuation)

    den_terms = []

    for cand_word, cand_ids in candidate_cache:
        cand_continuation = torch.cat([cand_ids, suffix_ids], dim=1)
        cand_lp = score_continuation_logprob(model, prefix_ids, cand_continuation)
        den_terms.append(cand_lp)

    # Ensure target included
    if target_word not in candidate_word_set:
        den_terms.append(log_num)

    log_den = torch.logsumexp(torch.tensor(den_terms, dtype=torch.float64), dim=0).item()

    bi_val = -(log_num - log_den) / math.log(2)

    if bi_val < -1e-6:
        print(f"[WARNING] Negative word-level BI surprisal: {bi_val:.6f}")

    return max(bi_val, 0.0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, default="data_output/bk21_stimuli_final.csv")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)

    parser.add_argument("--lexicon_file", type=str, required=True,
                        help="Path to candidate word lexicon (one word per line)")

    parser.add_argument("--output_col_uni", type=str, default="uni_surprisal_word")
    parser.add_argument("--output_col_bi", type=str, default="bi_surprisal_word")

    parser.add_argument("--prefix_col", type=str, default="prefix")
    parser.add_argument("--target_col", type=str, default="target_llm")
    parser.add_argument("--suffix_col", type=str, default="suffix")

    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    SMALL_MODEL_THRESHOLD = 1_000_000_000

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    config = AutoConfig.from_pretrained(args.model_id)
    with torch.device("meta"):
        dummy = AutoModelForCausalLM.from_config(config)
    n_params = sum(p.numel() for p in dummy.parameters())
    del dummy

    use_quantization = n_params >= SMALL_MODEL_THRESHOLD

    if use_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading model {args.model_id} ({n_params/1e9:.1f}B params) in 8-bit mode...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print(f"Loading model {args.model_id} ({n_params/1e6:.0f}M params) in fp32...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
        )

    device = next(model.parameters()).device
    model.eval()
    print(f"Model loaded on device: {device}")

    # Load and cache candidate words
    candidate_words = load_candidate_words(args.lexicon_file)
    candidate_word_set = set(candidate_words)
    print(f"Loaded {len(candidate_words)} candidate words from lexicon.")

    candidate_cache = build_candidate_cache(tokenizer, candidate_words, device)
    print(f"Cached {len(candidate_cache)} tokenized candidate words.")

    checkpoint_path = args.output_csv.replace(".csv", "_checkpoint.csv")
    metadata_path = args.output_csv.replace(".csv", "_metadata.pt")

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

    for i, row in tqdm(df.iterrows(), total=len(df)):
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

            bi_val = compute_bi_surprisal_word(
                model=model,
                tokenizer=tokenizer,
                prefix=row[args.prefix_col],
                target=row[args.target_col],
                suffix=row[args.suffix_col],
                candidate_cache=candidate_cache,
                candidate_word_set=candidate_word_set,
                device=device,
            )

            df.at[i, args.output_col_uni] = uni_val
            df.at[i, args.output_col_bi] = bi_val

            metadata_list.append({
                "index": i,
                "model_id": args.model_id,
                "uni_surprisal_word": uni_val,
                "bi_surprisal_word": bi_val,
                "token_data": res["token_data"] if res else None,
            })

            if i % args.save_every == 0:
                df.to_csv(checkpoint_path, index=False)
                torch.save(metadata_list, metadata_path)

        except KeyboardInterrupt:
            print(f"\nInterrupted at row {i}. Saving checkpoint...")
            df.to_csv(checkpoint_path, index=False)
            torch.save(metadata_list, metadata_path)
            print(f"Checkpoint saved to {checkpoint_path}. Re-run to resume.")
            return

        except Exception as e:
            print(f"Row {i} failed: {e}")
            traceback.print_exc()
            df.at[i, args.output_col_uni] = float("nan")
            df.at[i, args.output_col_bi] = float("nan")

    df.to_csv(args.output_csv, index=False)
    torch.save(metadata_list, metadata_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"Done! CSV saved to {args.output_csv}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()