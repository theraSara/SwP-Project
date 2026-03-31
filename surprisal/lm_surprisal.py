import argparse
import math
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def compute_uni_surprisal(model, tokenizer, prefix, target, device):
    """Calculates standard forward surprisal: P(target | prefix)"""
    if pd.isna(prefix) or pd.isna(target):
        return None

    prefix, target = str(prefix), str(target)
    # Ensure space handling
    full_text = prefix if prefix.endswith(" ") else prefix + " "
    full_text += target

    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    p_enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    
    p_len = p_enc.input_ids.shape[1]
    f_ids = enc.input_ids[0]
    f_len = f_ids.shape[1] if len(f_ids.shape) > 1 else f_ids.shape[0]

    with torch.no_grad():
        logits = model(**enc).logits[0]

    total_uni = 0.0
    for pos in range(p_len, f_len):
        token_id = f_ids[pos].item()
        # Logits at [pos-1] predict the token at [pos]
        log_probs = F.log_softmax(logits[pos - 1], dim=-1)
        total_uni += -log_probs[token_id].item() / math.log(2)

    return {"uni_val": total_uni}

def compute_bi_surprisal_batched(model, tokenizer, prefix, target, suffix, device, k=500, batch_size=20):
    """Calculates bidirectional surprisal: P(target | prefix, suffix) using Top-K batching"""
    if pd.isna(suffix) or str(suffix).strip() == "":
        return float("nan")
    
    p_ids = tokenizer.encode(str(prefix), return_tensors="pt", add_special_tokens=False).to(device)
    # Use only the first token of target for the bi-approx to keep it simple/standard
    t_ids = tokenizer.encode(str(target), return_tensors="pt", add_special_tokens=False).to(device)[:, :1]
    s_ids = tokenizer.encode(str(suffix), return_tensors="pt", add_special_tokens=False).to(device)
    
    target_id = t_ids[0, 0].item()
    p_len, s_len = p_ids.shape[1], s_ids.shape[1]

    # 1. Numerator
    full_seq = torch.cat([p_ids, t_ids, s_ids], dim=1)
    with torch.no_grad():
        logits = model(full_seq).logits[0]
    
    log_probs_at_target = F.log_softmax(logits[p_len - 1], dim=-1)
    log_prob_target = log_probs_at_target[target_id].item()
    
    log_prob_suffix = 0.0
    for i in range(s_len):
        step_lp = F.log_softmax(logits[p_len + i], dim=-1)
        log_prob_suffix += step_lp[s_ids[0, i]].item()
    
    log_num = log_prob_target + log_prob_suffix

    # 2. Denominator (Top-K candidates)
    top_k = torch.topk(log_probs_at_target, k)
    den_log_terms = []
    
    for i in range(0, k, batch_size):
        b_cands = top_k.indices[i : i + batch_size].unsqueeze(1)
        curr_b = b_cands.size(0)
        
        # Batch construct: [Prefix + Candidate + Suffix]
        b_full = torch.cat([p_ids.expand(curr_b, -1), b_cands, s_ids.expand(curr_b, -1)], dim=1)
        
        with torch.no_grad():
            b_logits = model(b_full).logits 

        for b_idx in range(curr_b):
            cand_log_prob_suffix = 0.0
            for s_pos in range(s_len):
                lp = F.log_softmax(b_logits[b_idx, p_len + s_pos], dim=-1)
                cand_log_prob_suffix += lp[s_ids[0, s_pos]].item()
            
            den_log_terms.append(top_k.values[i + b_idx].item() + cand_log_prob_suffix)

    log_den = torch.logsumexp(torch.tensor(den_log_terms), dim=0).item()
    return -(log_num - log_den) / math.log(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data_output/bk21_stimuli.csv")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_col_uni", type=str, default="uni_surprisal")
    parser.add_argument("--output_col_bi", type=str, default="bi_surprisal")
    parser.add_argument("--prefix_col", type=str, default="prefix")
    parser.add_argument("--target_col", type=str, default="target_llm")
    parser.add_argument("--suffix_col", type=str, default="suffix")
    parser.add_argument("--k", type=int, default=500, help="Top-K candidates for bidirectional approximation")
    args = parser.parse_args()

    df = pd.read_csv("data/bk21_stimuli.csv") 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Optimized for Mac
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()

    uni_results, bi_results = [], []

    # Use tqdm to see progress
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. Unidirectional
            res = compute_uni_surprisal(model, tokenizer, row['prefix'], row['target_llm'], device)
            # CRITICAL FIX: Match the key returned by the function
            uni_val = res["uni_val"] if res else float("nan")
            
            # 2. Bidirectional
            bi_val = compute_bi_surprisal_batched(model, tokenizer, row['prefix'], row['target_llm'], row['suffix'], device)

            uni_results.append(uni_val)
            bi_results.append(bi_val)

        except Exception as e:
            print(f"Row {i} failed with error: {e}")
            uni_results.append(float("nan"))
            bi_results.append(float("nan"))

    df["uni_surprisal"] = uni_results
    df["bi_surprisal"] = bi_results
    df.to_csv("bk21_results.csv", index=False)

if __name__ == "__main__":
    main()