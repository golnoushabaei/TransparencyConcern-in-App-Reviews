# src/stage1-data_curation/keyword_filtering_set1.py
import re
import pandas as pd
from .config_stage1 import Stage1Config

def find_matches_multiple_columns(row, keywords, columns):
    matched = set()
    for col in columns:
        text = str(row[col]).lower()
        matched.update(
            kw for kw in keywords
            if re.search(rf"\b{re.escape(kw.lower())}\b", text)
        )
    return list(matched)

def keyword_filter_set1(cfg: Stage1Config | None = None):
    cfg = cfg or Stage1Config()

    df = pd.read_csv(cfg.filtered_score_len)
    keywords_df = pd.read_csv(cfg.keywords_set1, header=None)  # one keyword per row
    keywords = keywords_df[0].dropna().tolist()

    df["matched_keywords"] = df.apply(
        lambda row: find_matches_multiple_columns(row, keywords, ["content"]),
        axis=1,
    )
    df_matched = df[df["matched_keywords"].str.len() > 0]

    cfg.filtered_keywords_set1.parent.mkdir(parents=True, exist_ok=True)
    df_matched.to_csv(cfg.filtered_keywords_set1, index=False)

    print(f"Filtered rows saved: {len(df_matched)}")
    print(f"Filtered file: {cfg.filtered_keywords_set1}")

    summary = []
    for kw in keywords:
        count = df_matched["matched_keywords"].apply(lambda x: kw in x).sum()
        summary.append((kw, count))

    summary_df = pd.DataFrame(summary, columns=["keyword", "row_count"])
    cfg.keyword_summary_set1.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(cfg.keyword_summary_set1, index=False)

    print(f"Keyword summary saved: {cfg.keyword_summary_set1}")

if __name__ == "__main__":
    keyword_filter_set1()