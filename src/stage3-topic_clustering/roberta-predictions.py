import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaModel


# ========= CONFIG =========

SEED = 42
pl.seed_everything(SEED, workers=True)

MODEL_NAME = "roberta-base"
MAX_LEN = 200
BATCH_SIZE = 128
CHUNK_SIZE = 100_000

# Paths (adjust in your README if needed)
GT_PATH = Path("data/ground_truth/Cleaned-Ground-Truth-DL.xlsx")
PARTS_GLOB = Path("data/intermediate/Final-Cleaned_Dataset_part*.csv")
CKPT_PATH = Path("models/roberta_new_gt_fold_6.ckpt")  # user-supplied path
OUTPUT_CHUNKS_DIR = Path("results/stage3/roberta_chunks")
FINAL_OUTPUT = Path("results/stage3/final_predictions.csv")


# ========= DATASET + MODEL =========

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)


class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class ReviewClassifier(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.pooler_output)

    def configure_optimizers(self):
        from torch.optim import AdamW
        return AdamW(self.parameters(), lr=self.lr)


# ========= PIPELINE =========

def load_and_filter_full_dataset():
    """
    - Load cleaned ground-truth (for de-duplication).
    - Load all cleaned parts (2.6M reviews split into multiple CSVs).
    - Remove rows whose clean_review appears in the ground truth.
    """
    gt_df = pd.read_excel(GT_PATH)
    gt_set = set(gt_df["clean_review"].astype(str))

    part_files = sorted(str(p) for p in PARTS_GLOB.parent.glob(PARTS_GLOB.name))
    if not part_files:
        raise FileNotFoundError(f"No files matched {PARTS_GLOB}")

    dfs = [pd.read_csv(f) for f in part_files]
    all_reviews_df = pd.concat(dfs, ignore_index=True)
    print("Original rows:", len(all_reviews_df))

    all_reviews_df["Review"] = all_reviews_df["Review"].astype(str)
    all_reviews_df["clean_review"] = all_reviews_df["clean_review"].astype(str)

    mask_not_in_gt = ~all_reviews_df["clean_review"].isin(gt_set)
    filtered_df = all_reviews_df[mask_not_in_gt].reset_index(drop=True)
    filtered_df["row_id"] = filtered_df.index

    print("Rows after removing ground truth:", len(filtered_df))
    return filtered_df


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_PATH}. "
            "Please set CKPT_PATH to your fine-tuned RoBERTa checkpoint."
        )

    model = ReviewClassifier.load_from_checkpoint(
        str(CKPT_PATH),
        lr=2e-5,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model, device


def run_predictions(filtered_df: pd.DataFrame, model, device: str):
    n_rows = len(filtered_df)
    n_chunks = math.ceil(n_rows / CHUNK_SIZE)
    print(f"Total rows: {n_rows}, chunks: {n_chunks}")

    OUTPUT_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min((chunk_idx + 1) * CHUNK_SIZE, n_rows)
        chunk_df = filtered_df.iloc[start:end].copy()
        print(f"\n=== Chunk {chunk_idx + 1}/{n_chunks}: rows {start}–{end - 1} ===")

        texts = chunk_df["clean_review"].tolist()
        infer_dataset = ReviewDataset(texts, tokenizer, MAX_LEN)
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        all_probs = []
        n_done = 0

        with torch.no_grad():
            for batch in infer_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits).squeeze(1)
                batch_probs = probs.cpu().numpy()
                all_probs.append(batch_probs)
                n_done += len(batch_probs)

                if n_done % 50_000 == 0 or n_done == len(texts):
                    print(f"  Chunk {chunk_idx + 1}: {n_done}/{len(texts)} records done")

        all_probs = np.concatenate(all_probs)
        all_preds = (all_probs >= 0.5).astype(int)

        chunk_df["pred_prob"] = all_probs
        chunk_df["pred_label"] = all_preds

        out_path = OUTPUT_CHUNKS_DIR / f"predictions_chunk_{chunk_idx + 1}.csv"
        cols = ["row_id", "Review", "clean_review", "pred_label", "pred_prob"]
        chunk_df[cols].to_csv(out_path, index=False)
        print(f"Saved {len(chunk_df)} rows to {out_path}")


def combine_chunks():
    chunk_files = sorted(OUTPUT_CHUNKS_DIR.glob("predictions_chunk_*.csv"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files in {OUTPUT_CHUNKS_DIR}")

    dfs = [pd.read_csv(f) for f in chunk_files]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values("row_id").reset_index(drop=True)

    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT, index=False)
    print("Final rows:", len(final_df))
    print(f"Final predictions saved to: {FINAL_OUTPUT}")


def main():
    filtered_df = load_and_filter_full_dataset()
    model, device = load_model()
    run_predictions(filtered_df, model, device)
    combine_chunks()


if __name__ == "__main__":
    main()