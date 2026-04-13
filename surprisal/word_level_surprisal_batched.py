import argparse
import math
import os
import traceback
import gc
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm


def safe_model_name(model_id):
    return model_id.replace("/", "_")


def score_continuation_logprob_single(model, prefix_ids, continuation_ids):
    full_ids = torch.cat([prefix_ids, continuation_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits[0]

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


def build_candidate_groups(tokenizer, candidate_words, device):
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


def score_candidate_batch(model, prefix_ids, cand_ids_batch, suffix_ids):
    B = cand_ids_batch.shape[0]
    p_len = prefix_ids.shape[1]
    cand_len = cand_ids_batch.shape[1]
    s_len = suffix_ids.shape[1]

    batch_prefix = prefix_ids.expand(B, -1)
    batch_suffix = suffix_ids.expand(B, -1)

    full_ids = torch.cat([batch_prefix, cand_ids_batch, batch_suffix], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits.float()

    total_lp = torch.zeros(B, device=logits.device)

    # candidate word tokens
    for j in range(cand_len):
        step_logits = logits[:, p_len - 1 + j, :]
        step_log_probs = F.log_softmax(step_logits, dim=-1)
        token_ids = cand_ids_batch[:, j].unsqueeze(1)
        total_lp += step_log_probs.gather(1, token_ids).squeeze(1)

    # suffix tokens
    for j in range(s_len):
        step_logits = logits[:, p_len + cand_len - 1 + j, :]
        step_log_probs = F.log_softmax(step_logits, dim=-1)
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
    if pd.isna(prefix) or pd.isna(target) or pd.isna(suffix):
        return float("nan")

    prefix_str = str(prefix).rstrip()
    target_word = str(target).strip()
    suffix_text = str(suffix).strip()
    suffix_str = (" " + suffix_text) if suffix_text else ""

    prefix_ids = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    suffix_ids = tokenizer(suffix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    target_ids = tokenizer(" " + target_word, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    true_continuation = torch.cat([target_ids, suffix_ids], dim=1)
    log_num = score_continuation_logprob_single(model, prefix_ids, true_continuation)

    den_terms = []

    for tok_len, group in candidate_groups.items():
        ids_list = group["ids"]
        n = len(ids_list)

        for start in range(0, n, cand_batch_size):
            end = min(start + cand_batch_size, n)
            batch_ids = ids_list[start:end]
            cand_ids_batch = torch.cat(batch_ids, dim=0)
            batch_lp = score_candidate_batch(model, prefix_ids, cand_ids_batch, suffix_ids)
            den_terms.append(batch_lp.detach().cpu())

    if target_word not in candidate_word_set:
        den_terms.append(torch.tensor([log_num], dtype=torch.float32))

    log_den = torch.logsumexp(torch.cat(den_terms).double(), dim=0).item()
    bi_val = -(log_num - log_den) / math.log(2)

    if bi_val < -1e-6:
        print(f"[WARNING] Negative word-level BI surprisal: {bi_val:.6f}")

    return max(bi_val, 0.0)


def load_model_and_tokenizer(model_id):
    SMALL_MODEL_THRESHOLD = 1_000_000_000

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        dummy = AutoModelForCausalLM.from_config(config)
    n_params = sum(p.numel() for p in dummy.parameters())
    del dummy

    use_quantization = n_params >= SMALL_MODEL_THRESHOLD

    if use_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading model {model_id} ({n_params/1e9:.1f}B params) in 8-bit mode...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
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


def cleanup_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_for_model(
    model_id,
    input_csv,
    output_csv,
    lexicon_file,
    output_col_uni,
    output_col_bi,
    prefix_col,
    target_col,
    suffix_col,
    cand_batch_size,
    save_every,
):
    print("=" * 80)
    print(f"RUNNING MODEL: {model_id}")
    print("=" * 80)

    df = pd.read_csv(input_csv)
    model, tokenizer, device = load_model_and_tokenizer(model_id)

    candidate_words = load_candidate_words(lexicon_file)
    print(f"Loaded {len(candidate_words)} candidate words.")

    candidate_groups, candidate_word_set = build_candidate_groups(tokenizer, candidate_words, device)
    print("Built candidate groups by token length:")
    for tok_len in sorted(candidate_groups.keys()):
        print(f"  token_len={tok_len}: {len(candidate_groups[tok_len]['words'])} words")

    checkpoint_path = output_csv.replace(".csv", "_checkpoint.csv")
    metadata_path = output_csv.replace(".csv", "_metadata.pt")

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)

        uni_ok = checkpoint_df[output_col_uni].notna()
        bi_ok = checkpoint_df[output_col_bi].notna()

        completed_indices = set(checkpoint_df.index[uni_ok & bi_ok])

        df[output_col_uni] = checkpoint_df[output_col_uni]
        df[output_col_bi] = checkpoint_df[output_col_bi]

        print(f"Resuming from checkpoint: {len(completed_indices)}/{len(df)} rows done.")
    else:
        completed_indices = set()
        df[output_col_uni] = float("nan")
        df[output_col_bi] = float("nan")

    if os.path.exists(metadata_path):
        all_meta = torch.load(metadata_path, weights_only=False)
        metadata_list = [m for m in all_meta if m["index"] in completed_indices]
    else:
        metadata_list = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{safe_model_name(model_id)}"):
        if i in completed_indices:
            continue

        try:
            res = compute_uni_surprisal_word(
                model=model,
                tokenizer=tokenizer,
                prefix=row[prefix_col],
                target=row[target_col],
                device=device,
            )
            uni_val = res["uni_val"] if res else float("nan")

            bi_val = compute_bi_surprisal_word_batched(
                model=model,
                tokenizer=tokenizer,
                prefix=row[prefix_col],
                target=row[target_col],
                suffix=row[suffix_col],
                candidate_groups=candidate_groups,
                candidate_word_set=candidate_word_set,
                device=device,
                cand_batch_size=cand_batch_size,
            )

            df.at[i, output_col_uni] = uni_val
            df.at[i, output_col_bi] = bi_val

            metadata_list.append({
                "index": i,
                "model_id": model_id,
                "uni_surprisal_word": uni_val,
                "bi_surprisal_word": bi_val,
                "token_data": res["token_data"] if res else None,
                "cand_batch_size": cand_batch_size,
                "lexicon_size": len(candidate_word_set),
            })

            if i % save_every == 0:
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
            df.at[i, output_col_uni] = float("nan")
            df.at[i, output_col_bi] = float("nan")

    df.to_csv(output_csv, index=False)
    torch.save(metadata_list, metadata_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"Done! CSV saved to {output_csv}")
    print(f"Metadata saved to {metadata_path}")

    cleanup_model(model, tokenizer)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, default="data_output/bk21_stimuli_final.csv")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_ids", type=str, nargs="+", required=True)
    parser.add_argument("--lexicon_file", type=str, required=True)

    parser.add_argument("--output_col_uni", type=str, default="uni_surprisal_word")
    parser.add_argument("--output_col_bi", type=str, default="bi_surprisal_word")

    parser.add_argument("--prefix_col", type=str, default="prefix")
    parser.add_argument("--target_col", type=str, default="target_llm")
    parser.add_argument("--suffix_col", type=str, default="suffix")

    parser.add_argument("--cand_batch_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_id in args.model_ids:
        model_name = safe_model_name(model_id)
        output_csv = os.path.join(args.output_dir, f"{model_name}_wordlevel_batched.csv")

        run_for_model(
            model_id=model_id,
            input_csv=args.input_csv,
            output_csv=output_csv,
            lexicon_file=args.lexicon_file,
            output_col_uni=args.output_col_uni,
            output_col_bi=args.output_col_bi,
            prefix_col=args.prefix_col,
            target_col=args.target_col,
            suffix_col=args.suffix_col,
            cand_batch_size=args.cand_batch_size,
            save_every=args.save_every,
        )


if __name__ == "__main__":
    main()