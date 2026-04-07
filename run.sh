#!/bin/bash

# GPT-2: full vocab, large batches
python surprisal/lm_surprisal.py \
  --input_csv data_output/bk21_stimuli_final.csv \
  --output_csv data_output/bk21_with_gpt2_v2.csv \
  --model_id gpt2 --k 50257 --batch_size 500

# Gemma-270m
python surprisal/lm_surprisal.py \
  --input_csv data_output/bk21_stimuli_final.csv \
  --output_csv data_output/bk21_with_gemma270m_v2.csv \
  --model_id google/gemma-3-270m --k 10000 --batch_size 20

# Gemma-12b
python surprisal/lm_surprisal.py \
  --input_csv data_output/bk21_stimuli_final.csv \
  --output_csv data_output/bk21_with_gemma12b_v2.csv \
  --model_id google/gemma-3-12b-pt --k 10000 --batch_size 20



run_model() {
    echo "Running $1..."
    python surprisal/lm_surprisal.py \
        --input_csv data_output/bk21_stimuli_final.csv \
        --output_csv "data_output/bk21_with_$2.csv" \
        --model_id "$1" \
        --output_col_uni "$2_uni_surprisal" \
        --output_col_bi "$2_bi_surprisal"
}

case $1 in
    "gpt2")
        run_model "gpt2" "gpt2"
        ;;
    "gemma270m")
        run_model "google/gemma-3-270m" "gemma270m"
        ;;
    "gemma12b")
        run_model "google/gemma-3-12b-pt" "gemma12b"
        ;;
    "all")
        run_model "gpt2" "gpt2"
        run_model "google/gemma-3-270m" "gemma270m"
        run_model "google/gemma-3-12b-pt" "gemma12b"
        ;;
    *)
        echo "Usage: ./run.sh [gpt2|gemma270m|gemma12b|all]"
        ;;
esac