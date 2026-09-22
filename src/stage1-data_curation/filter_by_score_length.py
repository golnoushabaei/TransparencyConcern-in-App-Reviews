# src/stage1-data_curation/filter_by_score_length.py
import pandas as pd
from .config_stage1 import Stage1Config

def filter_by_score_and_length(cfg: Stage1Config | None = None):
    cfg = cfg or Stage1Config()

    df = pd.read_csv(cfg.raw_data)
    filtered_df = df[
        (df["score"] <= 3) &
        (df["content"].str.split().str.len() > 4)
    ]

    cfg.filtered_score_len.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(cfg.filtered_score_len, index=False)

    print(f"Number of rows after filtering: {len(filtered_df)}")
    print(f"Saved to: {cfg.filtered_score_len}")

if __name__ == "__main__":
    filter_by_score_and_length()