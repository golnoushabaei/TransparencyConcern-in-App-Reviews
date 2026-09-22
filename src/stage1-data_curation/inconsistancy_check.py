"""Identify potentially inconsistent annotation pairs in app-review ground truth.

This script is intentionally non-generative and deterministic.
It only uses a pretrained sentence-transformer model (all-MiniLM-L6-v2) to generate embeddings and 
then computes cosine similarity between reviews.

Purpose:
    - Load the review dataset.
    - Preserve original row numbers.
    - Generate embeddings for all review texts.
    - Find pairs of reviews with high semantic similarity but different labels.
    - Export the flagged pairs for human review.
    - Report summary statistics.

Required packages:
    pip install pandas numpy sentence-transformers openpyxl

Usage:
    1. Edit the constants below: DATASET_PATH, REVIEW_TEXT_COLUMN, LABEL_COLUMN,
       SIMILARITY_THRESHOLD, OUTPUT_CSV_PATH.
    2. Run: python src/stage1-data_curation/inconsistancy_check.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
# File to load. Supports .csv or .xlsx/.xls
DATASET_PATH = Path("data/ground_truth/ground-truth-before-inconsistency.csv")

# Column names in the dataset
REVIEW_TEXT_COLUMN = "Review"
LABEL_COLUMN = "Label"

# Similarity threshold for identifying potentially inconsistent pairs.
# Set to 0.90 for a stricter 90% similarity cutoff.
SIMILARITY_THRESHOLD = 0.90

# Keep all mismatched pairs above the threshold.
# If you want a shortlist instead, set this to a small positive integer.
MAX_PAIRS_TO_EXPORT = None

# Output file for flagged pairs
OUTPUT_CSV_PATH = Path("results/stage1/potentially_inconsistent_pairs.csv")

# Embedding model (non-generative, deterministic embeddings only)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 64

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line options for rerunning the script from the terminal."""
    parser = argparse.ArgumentParser(
        description="Find semantically-similar reviews with conflicting labels for manual review.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/ground_truth/ground-truth-before-inconsistency.csv",
        help="Path to the CSV/Excel dataset to analyze.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/stage1/potentially_inconsistent_pairs.csv",
        help="Path to save the flagged review pairs.",
    )
    parser.add_argument(
        "--review-column",
        type=str,
        default="Review",
        help="Name of the review text column.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="Name of the binary label column.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Minimum cosine similarity threshold (for example 0.60 or 0.90).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on the number of highest-similarity pairs to export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset and validate the required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported dataset format: {path.suffix}. Use .csv, .xlsx, or .xls."
        )

    required_columns = {REVIEW_TEXT_COLUMN, LABEL_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Preserve original row number from the source file.
    df = df.copy()
    df["original_row_number"] = np.arange(len(df), dtype=int)
    return df


def normalize_labels(series: pd.Series) -> pd.Series:
    """Convert labels to numeric binary values (0/1), but do not alter content."""
    values = series.astype(str).str.strip()
    unique_values = sorted(set(values))

    # Keep binary labels as-is when they are already 0/1.
    normalized = values.copy()
    for current, target in {"0": 0, "1": 1, "False": 0, "True": 1}.items():
        normalized = normalized.str.replace(current, str(target), regex=False)

    # If labels are numeric strings, coerce to int.
    try:
        normalized = pd.to_numeric(normalized, errors="raise")
    except (TypeError, ValueError):
        pass

    # Validate binary labels.
    if not set(pd.Series(normalized).dropna().unique()).issubset({0, 1}):
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' must be binary with values 0 and 1. "
            f"Found: {sorted(set(pd.Series(normalized).dropna().unique()))}"
        )

    return pd.to_numeric(normalized, errors="coerce").astype("Int64")


def generate_embeddings(texts: Iterable[str], model_name: str, batch_size: int) -> np.ndarray:
    """Generate sentence embeddings using the pretrained sentence-transformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix from embeddings."""
    if embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_candidate_pairs(
    labels: pd.Series,
    similarity_matrix: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Return candidate review pairs (i, j, cosine_similarity) satisfying:
    - i != j
    - no duplicate pairs
    - labels differ
    - similarity >= threshold
    """
    n = len(labels)
    if n < 2:
        return []

    upper_tri = np.triu_indices(n, k=1)
    sim_values = similarity_matrix[upper_tri]

    keep_mask = (sim_values >= threshold) & (labels.to_numpy()[upper_tri[0]] != labels.to_numpy()[upper_tri[1]])
    pair_indices = np.where(keep_mask)[0]

    candidate_pairs: List[Tuple[int, int, float]] = []
    for idx in pair_indices:
        i = int(upper_tri[0][idx])
        j = int(upper_tri[1][idx])
        score = float(sim_values[idx])
        candidate_pairs.append((i, j, score))

    # Sort by descending similarity score for output.
    candidate_pairs.sort(key=lambda item: item[2], reverse=True)
    return candidate_pairs


def export_flagged_pairs(
    df: pd.DataFrame,
    candidate_pairs: List[Tuple[int, int, float]],
    output_path: Path,
) -> pd.DataFrame:
    """Create a DataFrame with review A/B metadata and cosine score."""
    if not candidate_pairs:
        output = pd.DataFrame(
            columns=[
                "original_row_number_review_a",
                "review_a_text",
                "review_a_label",
                "original_row_number_review_b",
                "review_b_text",
                "review_b_label",
                "cosine_similarity",
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        return output

    rows = []
    for i, j, score in candidate_pairs:
        a_row = df.iloc[i]
        b_row = df.iloc[j]

        rows.append(
            {
                "original_row_number_review_a": int(a_row["original_row_number"]),
                "review_a_text": str(a_row[REVIEW_TEXT_COLUMN]),
                "review_a_label": int(a_row[LABEL_COLUMN]),
                "original_row_number_review_b": int(b_row["original_row_number"]),
                "review_b_text": str(b_row[REVIEW_TEXT_COLUMN]),
                "review_b_label": int(b_row[LABEL_COLUMN]),
                "cosine_similarity": round(float(score), 8),
            }
        )

    output_df = pd.DataFrame(rows).sort_values(
        by="cosine_similarity",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_df


def print_summary(df: pd.DataFrame, flagged_pairs_df: pd.DataFrame, threshold: float, model_name: str) -> None:
    """Print summary statistics requested by the user."""
    total_reviews = len(df)
    label_zero = int((df[LABEL_COLUMN] == 0).sum())
    label_one = int((df[LABEL_COLUMN] == 1).sum())
    flagged_count = len(flagged_pairs_df)

    print("\n=== Potential inconsistency summary ===")
    print(f"Total number of reviews: {total_reviews}")
    print(f"Number of reviews labelled 0: {label_zero}")
    print(f"Number of reviews labelled 1: {label_one}")
    print(f"Number of potentially inconsistent pairs: {flagged_count}")
    print(f"Similarity threshold: {threshold}")
    print(f"Embedding model used: {model_name}")
    print("====================================\n")


def main() -> None:
    """Run the full deterministic inconsistency-check workflow."""
    args = parse_args()

    global DATASET_PATH, REVIEW_TEXT_COLUMN, LABEL_COLUMN, SIMILARITY_THRESHOLD
    global OUTPUT_CSV_PATH, EMBEDDING_BATCH_SIZE, MAX_PAIRS_TO_EXPORT

    DATASET_PATH = Path(args.input)
    REVIEW_TEXT_COLUMN = args.review_column
    LABEL_COLUMN = args.label_column
    SIMILARITY_THRESHOLD = args.threshold
    OUTPUT_CSV_PATH = Path(args.output)
    EMBEDDING_BATCH_SIZE = args.batch_size
    MAX_PAIRS_TO_EXPORT = args.max_pairs

    print(f"Loading dataset from: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)

    # Keep the exact textual values for manual review.
    texts = df[REVIEW_TEXT_COLUMN].fillna("").astype(str).tolist()

    print(f"Normalizing binary labels in '{LABEL_COLUMN}' column.")
    df[LABEL_COLUMN] = normalize_labels(df[LABEL_COLUMN])

    print(f"Generating embeddings with {EMBEDDING_MODEL_NAME}...")
    embeddings = generate_embeddings(texts, EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE)

    print("Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity_matrix(embeddings)

    print("Finding potentially inconsistent label pairs...")
    candidate_pairs = build_candidate_pairs(
        labels=df[LABEL_COLUMN],
        similarity_matrix=similarity_matrix,
        threshold=SIMILARITY_THRESHOLD,
    )

    # If MAX_PAIRS_TO_EXPORT is None, keep all qualifying pairs.
    if MAX_PAIRS_TO_EXPORT is not None:
        candidate_pairs = candidate_pairs[:MAX_PAIRS_TO_EXPORT]

    output_df = export_flagged_pairs(
        df=df,
        candidate_pairs=candidate_pairs,
        output_path=OUTPUT_CSV_PATH,
    )

    print_summary(
        df=df,
        flagged_pairs_df=output_df,
        threshold=SIMILARITY_THRESHOLD,
        model_name=EMBEDDING_MODEL_NAME,
    )

    if output_df.empty:
        print(f"No potentially inconsistent pairs found above the similarity threshold {SIMILARITY_THRESHOLD}.")
        print(f"Output file created at: {OUTPUT_CSV_PATH}")
    else:
        print(f"Flagged pairs saved to: {OUTPUT_CSV_PATH}")
        print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
