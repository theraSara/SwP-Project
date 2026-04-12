import argparse
import math
import os
import traceback
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm


def compute_uni_surprisal(model, tokenizer, prefix, target, device):
    if pd.isna(prefix) or pd.isna(target):
        return None

    prefix, target = str(prefix), str(target)
    target = " " + target.strip()

    p_enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(device)
    t_enc = tokenizer(target, return_tensors="pt", add_special_tokens=False).to(device)
    f_ids = torch.cat([p_enc.input_ids, t_enc.input_ids], dim=1)

    p_len = p_enc.input_ids.shape[1]
    f_len = f_ids.shape[1]

    with torch.no_grad():
        logits = model(f_ids).logits[0]

    total_uni = 0.0
    token_data = []

    for pos in range(p_len, f_len):
        token_id = f_ids[0, pos].item()

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

    return {"uni_val": total_uni, "token_data": token_data}


def _score_suffix_from_logits(batch_logits, p_len, s_ids):
    """
    batch_logits: [batch, seq_len, vocab]
    Returns:
        suffix log-prob for each batch item: [batch]
    """
    batch_size = batch_logits.shape[0]
    suffix_len = s_ids.shape[1]

    if suffix_len == 0:
        return torch.zeros(batch_size, device=batch_logits.device)

    suffix_logits = batch_logits[:, p_len:p_len + suffix_len, :].float()  # [B, suffix_len, V]
    suffix_log_probs = F.log_softmax(suffix_logits, dim=-1)

    gold_suffix = s_ids.expand(batch_size, -1).unsqueeze(-1)  # [B, suffix_len, 1]
    suffix_lp = suffix_log_probs.gather(2, gold_suffix).squeeze(-1).sum(dim=1)  # [B]

    return suffix_lp


def compute_bi_surprisal_topk(
    model,
    tokenizer,
    prefix,
    target,
    suffix,
    device,
    k=10000,
    cand_batch_size=128,
):
    if pd.isna(suffix) or str(suffix).strip() == "":
        return float("nan")

    target_str = " " + str(target).strip()
    prefix_str = str(prefix).rstrip()
    suffix_str = " " + str(suffix).lstrip()

    p_ids = tokenizer.encode(prefix_str, return_tensors="pt", add_special_tokens=True).to(device)
    t_ids = tokenizer.encode(target_str, return_tensors="pt", add_special_tokens=False).to(device)
    s_ids = tokenizer.encode(suffix_str, return_tensors="pt", add_special_tokens=False).to(device)

    total_bi_surprisal = 0.0
    current_p_ids = p_ids
    vocab_size = model.config.vocab_size

    for t_idx in range(t_ids.shape[1]):
        target_id = t_ids[0, t_idx].item()
        p_len = current_p_ids.shape[1]

        # Numerator: true token + suffix
        true_seq = torch.cat([current_p_ids, t_ids[:, t_idx:t_idx+1], s_ids], dim=1)

        with torch.no_grad():
            true_logits = model(true_seq).logits  # [1, seq, vocab]

        true_logits_0 = true_logits[0]

        target_log_probs = F.log_softmax(true_logits_0[p_len - 1].float(), dim=-1)
        log_prob_target = target_log_probs[target_id].item()

        log_prob_suffix = _score_suffix_from_logits(true_logits, p_len, s_ids)[0].item()
        log_num = log_prob_target + log_prob_suffix

        # Top-k candidate set
        k_eff = min(k, vocab_size)
        topk = torch.topk(target_log_probs, k_eff)
        cand_indices = topk.indices
        cand_lp = topk.values

        # ensure target included
        if not (cand_indices == target_id).any():
            cand_indices = torch.cat([cand_indices, torch.tensor([target_id], device=device)])
            cand_lp = torch.cat([cand_lp, target_log_probs[target_id].unsqueeze(0)])

        den_terms = []
        n_cands = cand_indices.shape[0]

        for start in range(0, n_cands, cand_batch_size):
            end = min(start + cand_batch_size, n_cands)
            curr_ids = cand_indices[start:end].unsqueeze(1)  # [bs, 1]
            curr_bs = curr_ids.shape[0]

            batch_prefix = current_p_ids.expand(curr_bs, -1)
            batch_suffix = s_ids.expand(curr_bs, -1)
            batch_seq = torch.cat([batch_prefix, curr_ids, batch_suffix], dim=1)

            with torch.no_grad():
                batch_logits = model(batch_seq).logits  # [bs, seq, vocab]

            suffix_lp = _score_suffix_from_logits(batch_logits, p_len, s_ids)
            terms = cand_lp[start:end] + suffix_lp
            den_terms.append(terms.detach().cpu())

        log_den = torch.logsumexp(torch.cat(den_terms).double(), dim=0).item()
        token_bi_val = -(log_num - log_den) / math.log(2)

        if token_bi_val < -1e-6:
            print(f"[WARNING] Negative top-k BI surprisal: {token_bi_val:.6f}")

        total_bi_surprisal += max(token_bi_val, 0.0)

        # advance with true token
        current_p_ids = torch.cat([current_p_ids, t_ids[:, t_idx:t_idx+1]], dim=1)

    return total_bi_surprisal


def compute_bi_surprisal_full_vocab(
    model,
    tokenizer,
    prefix,
    target,
    suffix,
    device,
    cand_batch_size=128,
):
    if pd.isna(suffix) or str(suffix).strip() == "":
        return float("nan")

    target_str = " " + str(target).strip()
    prefix_str = str(prefix).rstrip()
    suffix_str = " " + str(suffix).lstrip()

    p_ids = tokenizer.encode(prefix_str, return_tensors="pt", add_special_tokens=True).to(device)
    t_ids = tokenizer.encode(target_str, return_tensors="pt", add_special_tokens=False).to(device)
    s_ids = tokenizer.encode(suffix_str, return_tensors="pt", add_special_tokens=False).to(device)

    total_bi_surprisal = 0.0
    current_p_ids = p_ids
    vocab_size = model.config.vocab_size

    for t_idx in range(t_ids.shape[1]):
        target_id = t_ids[0, t_idx].item()
        p_len = current_p_ids.shape[1]

        # Numerator
        true_seq = torch.cat([current_p_ids, t_ids[:, t_idx:t_idx+1], s_ids], dim=1)

        with torch.no_grad():
            true_logits = model(true_seq).logits  # [1, seq, vocab]

        true_logits_0 = true_logits[0]

        target_log_probs = F.log_softmax(true_logits_0[p_len - 1].float(), dim=-1)
        log_prob_target = target_log_probs[target_id].item()

        log_prob_suffix = _score_suffix_from_logits(true_logits, p_len, s_ids)[0].item()
        log_num = log_prob_target + log_prob_suffix

        # Full-vocab denominator
        den_terms = []

        for start in range(0, vocab_size, cand_batch_size):
            end = min(start + cand_batch_size, vocab_size)
            curr_bs = end - start

            cand_ids = torch.arange(start, end, device=device).unsqueeze(1)  # [bs, 1]
            batch_prefix = current_p_ids.expand(curr_bs, -1)
            batch_suffix = s_ids.expand(curr_bs, -1)

            batch_seq = torch.cat([batch_prefix, cand_ids, batch_suffix], dim=1)

            with torch.no_grad():
                batch_logits = model(batch_seq).logits  # [bs, seq, vocab]

            cand_lp = target_log_probs[start:end]
            suffix_lp = _score_suffix_from_logits(batch_logits, p_len, s_ids)

            terms = cand_lp + suffix_lp
            den_terms.append(terms.detach().cpu())

        log_den = torch.logsumexp(torch.cat(den_terms).double(), dim=0).item()
        token_bi_val = -(log_num - log_den) / math.log(2)

        if token_bi_val < -1e-6:
            print(f"[WARNING] Negative full-vocab BI surprisal: {token_bi_val:.6f}")

        total_bi_surprisal += max(token_bi_val, 0.0)

        # advance with true token
        current_p_ids = torch.cat([current_p_ids, t_ids[:, t_idx:t_idx+1]], dim=1)

    return total_bi_surprisal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/bk21_stimuli.csv")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)

    parser.add_argument("--output_col_uni", type=str, default="uni_surprisal")
    parser.add_argument("--output_col_bi", type=str, default="bi_surprisal")

    parser.add_argument("--prefix_col", type=str, default="prefix")
    parser.add_argument("--target_col", type=str, default="target_llm")
    parser.add_argument("--suffix_col", type=str, default="suffix")

    parser.add_argument("--bi_mode", type=str, default="topk", choices=["topk", "full"])
    parser.add_argument("--k", type=int, default=10000,
                        help="Top-K candidates for bidirectional approximation")
    parser.add_argument("--cand_batch_size", type=int, default=128,
                        help="Candidate batch size for denominator scoring")

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
    print(f"Model loaded on device: {device}")
    model.eval()

    checkpoint_path = args.output_csv.replace(".csv", "_checkpoint.csv")
    metadata_path = args.output_csv.replace(".csv", "_metadata.pt")

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)

        uni_ok = checkpoint_df[args.output_col_uni].notna()
        bi_ok = checkpoint_df[args.output_col_bi].notna() & (checkpoint_df[args.output_col_bi] >= 0)

        completed_indices = set(checkpoint_df.index[uni_ok & bi_ok])

        df[args.output_col_uni] = checkpoint_df[args.output_col_uni]
        df[args.output_col_bi] = checkpoint_df[args.output_col_bi]

        n_negative = int((checkpoint_df[args.output_col_bi] < 0).sum())
        print(f"Resuming from checkpoint: {len(completed_indices)}/{len(df)} items done.")
        if n_negative > 0:
            print(f"  -> {n_negative} rows with negative BI values will be recomputed.")
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
            res = compute_uni_surprisal(
                model, tokenizer, row[args.prefix_col], row[args.target_col], device
            )
            uni_val = res["uni_val"] if res else float("nan")

            if args.bi_mode == "full":
                bi_val = compute_bi_surprisal_full_vocab(
                    model=model,
                    tokenizer=tokenizer,
                    prefix=row[args.prefix_col],
                    target=row[args.target_col],
                    suffix=row[args.suffix_col],
                    device=device,
                    cand_batch_size=args.cand_batch_size,
                )
            else:
                bi_val = compute_bi_surprisal_topk(
                    model=model,
                    tokenizer=tokenizer,
                    prefix=row[args.prefix_col],
                    target=row[args.target_col],
                    suffix=row[args.suffix_col],
                    device=device,
                    k=args.k,
                    cand_batch_size=args.cand_batch_size,
                )

            df.at[i, args.output_col_uni] = uni_val
            df.at[i, args.output_col_bi] = bi_val

            metadata_list.append({
                "index": i,
                "uni_surprisal": uni_val,
                "bi_surprisal": bi_val,
                "token_data": res["token_data"] if res else None,
                "model_id": args.model_id,
                "bi_mode": args.bi_mode,
                "k": args.k if args.bi_mode == "topk" else None,
                "cand_batch_size": args.cand_batch_size,
            })

            if i % 10 == 0:
                df.to_csv(checkpoint_path, index=False)
                torch.save(metadata_list, metadata_path)

        except KeyboardInterrupt:
            print(f"\nInterrupted at row {i}. Saving checkpoint...")
            df.to_csv(checkpoint_path, index=False)
            torch.save(metadata_list, metadata_path)
            print(f"Checkpoint saved to {checkpoint_path}. Re-run the same command to resume.")
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