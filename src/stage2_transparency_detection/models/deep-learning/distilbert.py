import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from pytorch_lightning.callbacks import ModelCheckpoint

# ========= CONFIG =========

SEED = 42
pl.seed_everything(SEED, workers=True)

DATA_PATH = Path("data/ground_truth/Cleaned-Ground-Truth-DL.xlsx")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 200
N_SPLITS = 10
MAX_EPOCHS = 5
BATCH_SIZE = 32

# ========= DATA =========

pd_data = pd.read_excel(DATA_PATH)
x_data = pd_data["clean_review"].tolist()
y_data = pd_data["Label"].astype(int).tolist()  # must be 0/1

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class ReviewDataModule(pl.LightningDataModule):
    def __init__(self, x_tr, y_tr, x_val, y_val, x_te, y_te, tokenizer, batch_size=BATCH_SIZE):
        super().__init__()
        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_val, self.y_val = x_val, y_val
        self.x_te, self.y_te = x_te, y_te
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = ReviewDataset(self.x_tr, self.y_tr, self.tokenizer, MAX_LEN)
        self.val_ds = ReviewDataset(self.x_val, self.y_val, self.tokenizer, MAX_LEN)
        self.test_ds = ReviewDataset(self.x_te, self.y_te, self.tokenizer, MAX_LEN)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=3)


class ReviewClassifier(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler; use [CLS] token (position 0)
        return self.classifier(out.last_hidden_state[:, 0])

    def training_step(self, batch, batch_idx):
        labels = batch["label"].unsqueeze(1)
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"].unsqueeze(1)
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        labels = batch["label"].unsqueeze(1)
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, labels)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


def main():
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x_data)):
        print(f"\n===== FOLD {fold + 1} =====")

        x_train = [x_data[i] for i in train_idx]
        y_train = [y_data[i] for i in train_idx]
        x_test = [x_data[i] for i in test_idx]
        y_test = [y_data[i] for i in test_idx]

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=SEED
        )

        dm = ReviewDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, tokenizer)
        dm.setup()

        model = ReviewClassifier()

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"distilbert_new_gt_fold_{fold+1}",
        )

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            logger=False,
        )

        trainer.fit(model, dm)

        model.eval()
        probs, labels = [], []

        with torch.no_grad():
            for batch in dm.test_dataloader():
                logits = model(batch["input_ids"], batch["attention_mask"])
                prob = torch.sigmoid(logits).squeeze()
                probs.append(prob.cpu().numpy())
                labels.append(batch["label"].cpu().numpy())

        y_prob = np.concatenate(probs)
        y_true = np.concatenate(labels)
        y_pred = (y_prob >= 0.5).astype(int)

        print("Precision:", metrics.precision_score(y_true, y_pred))
        print("Recall:", metrics.recall_score(y_true, y_pred))
        print("F1:", metrics.f1_score(y_true, y_pred))
        print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
        print("Best model saved at:", checkpoint_callback.best_model_path)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        del model, dm, probs, labels
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== FINAL RESULTS =====")
    print("F1:", metrics.f1_score(all_true, all_pred))
    print("Accuracy:", metrics.accuracy_score(all_true, all_pred))
    print("Precision:", metrics.precision_score(all_true, all_pred))
    print("Recall:", metrics.recall_score(all_true, all_pred))


if __name__ == "__main__":
    main()