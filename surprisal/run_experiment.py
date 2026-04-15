import os
import argparse
import traceback

import torch
import pandas as pd
from tqdm import tqdm

from io_utils import (
    finalize_run,
    load_checkpoint,
    load_metadata,
    make_output_paths,
    save_progress,
)
from lexicon_utils import build_candidate_groups, build_candidate_words
from model_utils import (
    cleanup_model,
    load_model_and_tokenizer,
    safe_model_name,
    safe_mode_name,
)
from scoring import (
    compute_bi_surprisal_word_batched,
    compute_uni_surprisal_word,
)


def run_for_model_and_mode(model_id, lexicon_mode, args):
    print("=" * 90)
    print(f"RUNNING MODEL: {model_id}")
    print(f"LEXICON MODE : {lexicon_mode}")
    print("=" * 90)

    df = pd.read_csv(args.input_csv)

    model, tokenizer, device = load_model_and_tokenizer(
        model_id,
        force_no_quant=args.no_quantization,
    )

    candidate_words = build_candidate_words(
        mode=lexicon_mode,
        df=df,
        target_col=args.target_col,
        lexicon_file=args.lexicon_file,
        wordfreq_topn=args.wordfreq_topn,
    )
    print(f"Loaded {len(candidate_words)} candidate words for mode={lexicon_mode}")

    candidate_groups, candidate_word_set = build_candidate_groups(
        tokenizer, candidate_words, device
    )

    print("Built candidate groups by token length:")
    for tok_len in sorted(candidate_groups.keys()):
        print(f"  token_len={tok_len}: {len(candidate_groups[tok_len]['words'])} words")

    model_name = safe_model_name(model_id)
    mode_name = safe_mode_name(lexicon_mode)

    output_csv, checkpoint_path, metadata_path = make_output_paths(
        args.output_dir, model_name, mode_name
    )

    df, completed_indices = load_checkpoint(
        df, checkpoint_path, args.output_col_uni, args.output_col_bi
    )
    metadata_list = load_metadata(metadata_path, completed_indices)

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
                save_progress(df, checkpoint_path, metadata_list, metadata_path)

        except KeyboardInterrupt:
            print(f"\nInterrupted at row {i}. Saving checkpoint...")
            save_progress(df, checkpoint_path, metadata_list, metadata_path)
            print(f"Checkpoint saved to {checkpoint_path}. Re-run to resume.")
            cleanup_model(model, tokenizer)
            return

        except Exception as e:
            print(f"Row {i} failed: {e}")
            traceback.print_exc()
            df.at[i, args.output_col_uni] = float("nan")
            df.at[i, args.output_col_bi] = float("nan")

    finalize_run(df, output_csv, metadata_list, metadata_path, checkpoint_path)
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
    parser.add_argument("--lexicon_file", type=str, default=None)

    parser.add_argument("--output_col_uni", type=str, default="uni_surprisal_word")
    parser.add_argument("--output_col_bi", type=str, default="bi_surprisal_word")

    parser.add_argument("--prefix_col", type=str, default="prefix")
    parser.add_argument("--target_col", type=str, default="target_llm")
    parser.add_argument("--suffix_col", type=str, default="suffix")

    parser.add_argument("--cand_batch_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--wordfreq_topn", type=int, default=100000)
    parser.add_argument("--no_quantization", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_id in args.model_ids:
        for lexicon_mode in args.lexicon_modes:
            run_for_model_and_mode(model_id, lexicon_mode, args)


if __name__ == "__main__":
    main()