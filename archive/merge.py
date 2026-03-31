import pandas as pd

trial = pd.read_csv("data/bk21_clean_trial_level.csv")
surp = pd.read_csv("data/bk21_gpt2_surprisal.csv")

df = trial.merge(
    surp[["ITEM","condition","gpt2_s_surprisal"]],
    on=["ITEM","condition"],
    how="left"
)

df.to_csv("data/bk21_ready_for_model.csv", index=False)
