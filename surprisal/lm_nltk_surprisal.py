import argparse
import math
import os
import traceback
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm
import nltk

def compute_uni_surprisal(model, tokenizer, prefix, target, device, allowed_tokens):
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
        step_logits = logits[pos - 1].to(torch.float32)

        # --- NLTK VOCABULARY MASK ---
        masked_logits = torch.full_like(step_logits, float('-inf'))
        masked_logits[allowed_tokens] = step_logits[allowed_tokens]
        masked_logits[token_id] = step_logits[token_id] # Always allow true target so it doesn't crash on plurals/punctuation

        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)

        token_surprisal = -log_probs[token_id].item() / math.log(2)
        total_uni += token_surprisal

        entropy_bits = -torch.sum(probs * (log_probs / math.log(2))).item()

        token_data.append({
            "token_id": token_id,
            "token_str": tokenizer.decode([token_id]),
            "surprisal": token_surprisal,
            "entropy": entropy_bits,
            "logits": masked_logits.cpu().numpy()
        })

    return {"uni_val": total_uni, "token_data": token_data}


def compute_bi_surprisal_batched(model, tokenizer, prefix, target, suffix, device, allowed_tokens,
                                  k=500, batch_size=20, full_vocab_threshold=60000):
    if pd.isna(suffix) or str(suffix).strip() == "":
        return float("nan")

    target_str = " " + str(target).strip()
    prefix_str = str(prefix).rstrip()
    suffix_str = " " + str(suffix).lstrip()

    p_ids = tokenizer.encode(prefix_str, return_tensors="pt", add_special_tokens=True).to(device)
    t_enc = tokenizer.encode(target_str, return_tensors="pt", add_special_tokens=False).to(device)
    s_ids = tokenizer.encode(suffix_str, return_tensors="pt", add_special_tokens=False).to(device)
    s_len = s_ids.shape[1]

    total_bi_surprisal = 0.0
    current_p_ids = p_ids

    for t_idx in range(t_enc.shape[1]):
        target_id = t_enc[0, t_idx].item()
        p_len = current_p_ids.shape[1]

        full_seq = torch.cat([current_p_ids, t_enc[:, t_idx:t_idx+1], s_ids], dim=1)
        with torch.no_grad():
            logits = model(full_seq).logits[0]

        # --- NLTK VOCABULARY MASK ---
        step_logits = logits[p_len - 1].to(torch.float32)
        masked_logits = torch.full_like(step_logits, float('-inf'))
        masked_logits[allowed_tokens] = step_logits[allowed_tokens]
        masked_logits[target_id] = step_logits[target_id] # Force allow true target

        log_probs_at_target = F.log_softmax(masked_logits, dim=-1)
        log_prob_target = log_probs_at_target[target_id].item()

        # Suffix probabilities (we don't mask this so it can predict punctuation normally)
        log_prob_suffix = 0.0
        for i in range(s_len):
            step_lp = F.log_softmax(logits[p_len + i].to(torch.float32), dim=-1)
            log_prob_suffix += step_lp[s_ids[0, i]].item()

        log_num = log_prob_target + log_prob_suffix

        # Denominator selection - Because we masked with -inf, Top-K will naturally ONLY pull NLTK words!
        vocab_size = logits.shape[-1]
        
        # If the number of allowed NLTK tokens is smaller than K, we just evaluate all allowed tokens!
        actual_k = min(k, len(allowed_tokens))
        top_k = torch.topk(log_probs_at_target, actual_k)
        
        cand_indices = top_k.indices
        cand_log_probs = top_k.values

        if not (cand_indices == target_id).any():
            cand_indices = torch.cat([cand_indices, torch.tensor([target_id], device=device)])
            target_lp = log_probs_at_target[target_id].unsqueeze(0)
            cand_log_probs = torch.cat([cand_log_probs, target_lp])

        den_log_terms = []
        n_cands = cand_indices.shape[0]

        for i in range(0, n_cands, batch_size):
            b_indices = cand_indices[i: i + batch_size]
            curr_batch = b_indices.size(0)

            b_cands = b_indices.unsqueeze(1)
            b_full = torch.cat([
                current_p_ids.expand(curr_batch, -1),
                b_cands,
                s_ids.expand(curr_batch, -1)
            ], dim=1)

            with torch.no_grad():
                b_logits = model(b_full).logits

            for b_idx in range(curr_batch):
                cand_log_prob_suffix = 0.0
                for s_pos in range(s_len):
                    lp = F.log_softmax(b_logits[b_idx, p_len + s_pos].to(torch.float32), dim=-1)
                    cand_log_prob_suffix += lp[s_ids[0, s_pos]].item()

                den_log_terms.append(cand_log_probs[i + b_idx].item() + cand_log_prob_suffix)

        log_den = torch.logsumexp(
            torch.tensor(den_log_terms, dtype=torch.float64), 
            dim=0
        ).item()

        token_bi_val = -(log_num - log_den) / math.log(2)

        NOISE_THRESHOLD = 0.5
        if token_bi_val < 0:
            if abs(token_bi_val) <= NOISE_THRESHOLD:
                token_bi_val = 0.0
            else:
                print(f"  [WARNING] large negative bi={token_bi_val:.4f} bits — possible real error. "
                      f"log_num={log_num:.3f}, log_den={log_den:.3f}, k_used={n_cands})")

        total_bi_surprisal += token_bi_val
        current_p_ids = torch.cat([current_p_ids, t_enc[:, t_idx:t_idx+1]], dim=1)

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
    parser.add_argument("--k", type=int, default=5000,
                        help="Top-K candidates for bidirectional approximation")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size for denominator forward passes")
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
            args.model_id, quantization_config=bnb_config, device_map="auto"
        )
    else:
        print(f"Loading model {args.model_id} ({n_params/1e6:.0f}M params) in fp32...")
        model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    model.eval()

    # --- NLTK VOCABULARY SETUP ---
    print("Setting up human vocabulary mask (NLTK)...")
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        print("Downloading NLTK words list...")
        nltk.download('words')
    
    from nltk.corpus import words
    nltk_set = set(w.lower() for w in words.words())
    
    allowed_token_ids = []
    vocab_size = tokenizer.vocab_size
    
    # Check every token in the model's vocabulary
    for i in range(vocab_size):
        try:
            tok_str = tokenizer.decode([i]).strip().lower()
            # If the decoded token is a valid English word, allow it
            if tok_str in nltk_set and tok_str.isalpha():
                allowed_token_ids.append(i)
        except:
            pass
            
    allowed_tokens_tensor = torch.tensor(allowed_token_ids, device=device)
    print(f"Filtered LLM vocabulary from {vocab_size} down to {len(allowed_tokens_tensor)} valid English words.")

    checkpoint_path = args.output_csv.replace(".csv", "_checkpoint.csv")

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)

        uni_ok = checkpoint_df[args.output_col_uni].notna()
        bi_ok = checkpoint_df[args.output_col_bi].notna() & \
                (checkpoint_df[args.output_col_bi] >= 0)

        completed_indices = set(checkpoint_df.index[uni_ok & bi_ok])

        df[args.output_col_uni] = checkpoint_df[args.output_col_uni]
        df[args.output_col_bi] = checkpoint_df[args.output_col_bi]

        n_negative = int((checkpoint_df[args.output_col_bi] < 0).sum())
        print(f"Resuming from checkpoint: {len(completed_indices)}/{len(df)} items done.")
        if n_negative > 0:
            print(f"  → {n_negative} rows with negative bi will be recomputed.")
    else:
        completed_indices = set()
        df[args.output_col_uni] = float("nan")
        df[args.output_col_bi] = float("nan")

    metadata_path = args.output_csv.replace(".csv", "_metadata.pt")
    if os.path.exists(metadata_path):
        all_meta = torch.load(metadata_path, weights_only=False)
        metadata_list = [m for m in all_meta if m['index'] in completed_indices]
    else:
        metadata_list = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i in completed_indices:
            continue
        try:
            res = compute_uni_surprisal(
                model, tokenizer, row[args.prefix_col], row[args.target_col], device, allowed_tokens_tensor
            )
            uni_val = res["uni_val"] if res else float("nan")
            
            bi_val = compute_bi_surprisal_batched(
                model, tokenizer,
                row[args.prefix_col], row[args.target_col], row[args.suffix_col],
                device, allowed_tokens=allowed_tokens_tensor, k=args.k, batch_size=args.batch_size
            )

            df.at[i, args.output_col_uni] = uni_val
            df.at[i, args.output_col_bi] = bi_val

            if res:
                metadata_list.append({
                    "index": i,
                    "uni_surprisal": uni_val,
                    "bi_surprisal": bi_val,
                    "token_data": res["token_data"],
                    "model_id": args.model_id
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