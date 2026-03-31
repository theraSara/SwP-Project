import pandas as pd
import numpy as np

rt = pd.read_csv("data/SPRT_LogLin_216.csv")
stim = pd.read_csv("data/bk21_stimuli.csv")

rt["condition"] = rt["condition"].str.upper()

df = rt.merge(stim, on=["ITEM", "condition"], how="left")

print("Columns after merge:")
print(df.columns)

df = df.rename(columns={"critical_word_y": "critical_word"})

df["log_rt"] = np.log(df["SUM_3RT_trimmed"])

df = df[[
    "SUB",
    "ITEM",
    "condition",
    "sentence",
    "critical_word",
    "SUM_3RT_trimmed",
    "log_rt"
]]

df.to_csv("data/bk21_clean_trial_level.csv", index=False)

print("Final shape:", df.shape)