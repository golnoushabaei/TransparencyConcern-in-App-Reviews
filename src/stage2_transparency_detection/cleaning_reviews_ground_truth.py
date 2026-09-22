import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


# ========= CONFIG =========

# Input: ground truth after inconsistency removal
INPUT_PATH = Path("data/ground_truth/ground-truth-final.csv")

# Output: cleaned ground truth used for DL models
OUTPUT_PATH = Path("results/ground_truth/Cleaned-Ground-Truth-DL.xlsx")


# ========= PREPROCESSING FUNCTIONS =========

def pre_process(text: str) -> str:
    """
    - Remove HTML
    - Keep only alphabetic characters
    - Convert to lowercase
    - Remove extra whitespaces
    """
    # Remove HTML tags
    text = BeautifulSoup(str(text), "html.parser").get_text()
    # Keep only letters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    tokens = text.split()
    return " ".join(tokens)


def pre_process_v2(text: str) -> str:
    """
    - Remove single-character tokens
    """
    tokens = str(text).split()
    tokens = [token for token in tokens if len(token) > 1]
    return " ".join(tokens)


def main():
    # -----------------------------
    # Load the dataset
    # -----------------------------
    df = pd.read_csv(INPUT_PATH)  # change to read_excel if your source is xlsx

    # -----------------------------
    # Clean the Review text
    # -----------------------------
    df["clean_review"] = df["Review"].apply(pre_process)
    df["clean_review"] = df["clean_review"].apply(pre_process_v2)

    # -----------------------------
    # Remove very short reviews
    # (keep reviews with > 3 words)
    # -----------------------------
    df = df[df["clean_review"].apply(lambda x: len(x.split()) > 3)]

    # -----------------------------
    # Ensure labels are integers
    # -----------------------------
    df["Label"] = df["Label"].astype(int)

    # -----------------------------
    # Reset index after filtering
    # -----------------------------
    df.reset_index(drop=True, inplace=True)

    # -----------------------------
    # Save cleaned dataset
    # -----------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(
        OUTPUT_PATH,
        index=False,
        header=True,
        encoding="utf-8",
    )
    print(f"Preprocessing complete. File saved as {OUTPUT_PATH}")


if __name__ == "__main__":
    main()