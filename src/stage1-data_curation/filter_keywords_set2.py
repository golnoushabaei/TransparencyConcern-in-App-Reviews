# src/stage1-data_curation/keyword_filtering_set2.py
import re
import pandas as pd
from .config_stage1 import Stage1Config

def find_first_match(text, keywords):
    text = str(text).lower()
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", text):
            return kw
    return None

def keyword_filter_set2(cfg: Stage1Config | None = None):
    cfg = cfg or Stage1Config()

    data_file = cfg.filtered_score_len
    keywords_file = cfg.keywords_set2
    output_filtered = cfg.filtered_keywords_set2
    output_summary = cfg.keyword_summary_set2

    keywords_df = pd.read_csv(keywords_file)
    keywords = [kw.lower() for kw in keywords_df["Keyword"].dropna()]

    summary_counts = {kw: 0 for kw in keywords}

    chunksize = 10000
    first_chunk = True
    chunk_num = 0

    output_filtered.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    for chunk in pd.read_csv(data_file, chunksize=chunksize):
        chunk_num += 1
        print(f"Processing chunk {chunk_num}...")

        chunk["matched_keywords"] = chunk["content"].apply(
            lambda text: find_first_match(text, keywords)
        )

        matched_chunk = chunk[chunk["matched_keywords"].notna()]

        if not matched_chunk.empty:
            matched_chunk.to_csv(
                output_filtered,
                mode="a",
                index=False,
                header=first_chunk,
            )
            first_chunk = False

            for kw in matched_chunk["matched_keywords"]:
                summary_counts[kw] += 1

    summary_df = pd.DataFrame(
        list(summary_counts.items()), columns=["keyword", "row_count"]
    )
    summary_df.to_csv(output_summary, index=False)

    print("Filtered rows saved to:", output_filtered)
    print("Keyword summary saved to:", output_summary)

if __name__ == "__main__":
    keyword_filter_set2()