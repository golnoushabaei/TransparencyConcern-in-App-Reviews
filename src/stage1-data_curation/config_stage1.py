# config_stage1.py
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Stage1Config:
    raw_data: Path = Path("data/raw/RawData.csv")
    filtered_score_len: Path = Path("data/intermediate/DataFilteredByScore.csv")
    keywords_set1: Path = Path("data/keywords/keywords_set1.csv")
    keywords_set2: Path = Path("data/keywords/keywords_set2.csv")
    filtered_keywords_set1: Path = Path("data/intermediate/FilteredData_keywords_set1.csv")
    filtered_keywords_set2: Path = Path("data/intermediate/FilteredData_keywords_set2.csv")
    keyword_summary_set1: Path = Path("results/stage1/Keyword_summary_set1.csv")
    keyword_summary_set2: Path = Path("results/stage1/Keyword_summary_set2.csv")