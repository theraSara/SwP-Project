import pandas as pd

df = pd.read_csv("data/bk21_clean_trial_level.csv")

stim_unique = df[["ITEM", "condition", "sentence", "critical_word"]].drop_duplicates()

print("Unique rows:", len(stim_unique))  # should be 648

stim_unique.to_csv("data/bk21_unique_sentences.csv", index=False)