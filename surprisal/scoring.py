import math
import pandas as pd
import torch
import torch.nn.functional as F


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