

# GPT-2
python surprisal/lm_surprisal.py --input_csv data/bk21_stimuli.csv --output_csv data_output/bk21_with_gpt2.csv --model_id gpt2 --output_col_uni gpt2_uni_surprisal --output_col_bi gpt2_bi_surprisal

# Gemma 270m
python surprisal/lm_surprisal.py --input_csv data/bk21_stimuli.csv --output_csv data_output/bk21_with_gemma270m.csv --model_id google/gemma-3-270m --output_col_uni gemma270m_uni_surprisal --output_col_bi gemma270m_bi_surprisal

# Gemma 12pt
python surprisal/lm_surprisal.py --input_csv data/bk21_stimuli.csv --output_csv data_output/bk21_with_gemma12b.csv --model_id google/gemma-3-12b-pt --output_col_uni gemma12b_uni_surprisal --output_col_bi gemma12b_bi_surprisal
