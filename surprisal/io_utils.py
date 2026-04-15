import os
import pandas as pd
import torch


def make_output_paths(output_dir, model_name, mode_name):
    output_csv = os.path.join(output_dir, f"{model_name}__{mode_name}__wordlevel.csv")
    checkpoint_path = output_csv.replace(".csv", "_checkpoint.csv")
    metadata_path = output_csv.replace(".csv", "_metadata.pt")
    return output_csv, checkpoint_path, metadata_path


def load_checkpoint(df, checkpoint_path, output_col_uni, output_col_bi):
    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)

        uni_ok = checkpoint_df[output_col_uni].notna()
        bi_ok = checkpoint_df[output_col_bi].notna()

        completed_indices = set(checkpoint_df.index[uni_ok & bi_ok])

        df[output_col_uni] = checkpoint_df[output_col_uni]
        df[output_col_bi] = checkpoint_df[output_col_bi]

        print(f"Resuming from checkpoint: {len(completed_indices)}/{len(df)} rows done.")
    else:
        completed_indices = set()
        df[output_col_uni] = float("nan")
        df[output_col_bi] = float("nan")

    return df, completed_indices


def load_metadata(metadata_path, completed_indices):
    if os.path.exists(metadata_path):
        all_meta = torch.load(metadata_path, weights_only=False)
        return [m for m in all_meta if m["index"] in completed_indices]
    return []


def save_progress(df, checkpoint_path, metadata_list, metadata_path):
    df.to_csv(checkpoint_path, index=False)
    torch.save(metadata_list, metadata_path)


def finalize_run(df, output_csv, metadata_list, metadata_path, checkpoint_path):
    df.to_csv(output_csv, index=False)
    torch.save(metadata_list, metadata_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"Done! CSV saved to {output_csv}")
    print(f"Metadata saved to {metadata_path}")