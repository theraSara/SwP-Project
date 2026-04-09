#!/bin/bash

# Ensure the output directory exists
mkdir -p data_output

run_model() {
    MODEL_ID=\$1
    NAME=\$2
    K_VAL=\$3
    BATCH_SIZE=\$4

    echo "Starting $NAME ($MODEL_ID)"
    echo "Parameters: Top-K = $K_VAL | Batch Size = $BATCH_SIZE"

    python surprisal/lm_nltk_surprisal.py \
        --input_csv data_output/bk21_stimuli_final.csv \
        --output_csv "data_output/bk21_with_${NAME}_nltk.csv" \
        --model_id "$MODEL_ID" \
        --k "$K_VAL" \
        --batch_size "$BATCH_SIZE" \
        --output_col_uni "uni_surprisal" \
        --output_col_bi "bi_surprisal"
}

case \$1 in
    "gpt2")
        # GPT-2: Exact math (k is ignored due to small vocab), massive batch size
        run_model "gpt2" "gpt2" 50000 500
        ;;
    "gemma270m")
        # Gemma 270M: Easy for 32GB VRAM, batch size 100 runs incredibly fast
        run_model "google/gemma-3-270m" "gemma270m" 10000 100
        ;;
    "gemma12b")
        # Gemma 12B: Fits well in 32GB VRAM, batch size 200 speeds it up massively
        run_model "google/gemma-3-12b-pt" "gemma12b" 10000 200
        ;;
    "all")
        echo "Running ALL models sequentially..."
        run_model "gpt2" "gpt2" 50000 500
        run_model "google/gemma-3-270m" "gemma270m" 10000 100
        run_model "google/gemma-3-12b-pt" "gemma12b" 10000 200
        echo "All models finished!"
        ;;
    *)
        echo "Invalid command."
        echo "Usage: ./run_nltk.sh [gpt2|gemma270m|gemma12b|all]"
        ;;
esac