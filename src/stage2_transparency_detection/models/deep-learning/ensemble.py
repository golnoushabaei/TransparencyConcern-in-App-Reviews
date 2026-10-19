import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from pytorch_lightning.callbacks import ModelCheckpoint


# ========= CONFIG =========

SEED = 42
pl.seed_everything(SEED, workers=True)

DATA_PATH = Path("data/ground_truth/Cleaned-Ground-Truth-DL.xlsx")

MAX_LEN = 200
N_SPLITS = 5
MAX_EPOCHS = 5
BATCH_SIZE = 32

BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
DISTIL_MODEL_NAME = "distilbert-base-uncased"


# ========= DATA =========

pd_data = pd.read_excel(DATA_PATH)
x_data = pd_data["clean_review"].tolist()
y_data = pd_data["Label"].astype(int).tolist()  # must be 0/1

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
distil_tokenizer = AutoTokenizer.from_pretrained(DISTIL_MODEL_NAME)


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
    def __init__(self, model_name, lr=2e-5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(pooled)

    def training_step(self, batch, batch_idx):
        labels = batch["label"].unsqueeze(1)
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
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


def get_probs(model, dataloader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            prob = torch.sigmoid(logits).squeeze()
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs)


def main():
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_true, all_pred_majority, all_pred_unanimous = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x_data)):
        print(f"\n===== FOLD {fold + 1} =====")

        x_train = [x_data[i] for i in train_idx]
        y_train = [y_data[i] for i in train_idx]
        x_test = [x_data[i] for i in test_idx]
        y_test = [y_data[i] for i in test_idx]

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=SEED
        )

        dm_bert = ReviewDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, bert_tokenizer)
        dm_roberta = ReviewDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, roberta_tokenizer)
        dm_distil = ReviewDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, distil_tokenizer)

        dm_bert.setup()
        dm_roberta.setup()
        dm_distil.setup()

        bert_model = ReviewClassifier(BERT_MODEL_NAME)
        roberta_model = ReviewClassifier(ROBERTA_MODEL_NAME)
        distil_model = ReviewClassifier(DISTIL_MODEL_NAME)

        checkpoint_callback_bert = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"ensemble_bert_fold_{fold+1}",
        )
        checkpoint_callback_roberta = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"ensemble_roberta_fold_{fold+1}",
        )
        checkpoint_callback_distil = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"ensemble_distilbert_fold_{fold+1}",
        )

        trainer_bert = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback_bert],
            enable_checkpointing=True,
            logger=False,
        )
        trainer_roberta = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback_roberta],
            enable_checkpointing=True,
            logger=False,
        )
        trainer_distil = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback_distil],
            enable_checkpointing=True,
            logger=False,
        )

        trainer_bert.fit(bert_model, dm_bert)
        trainer_roberta.fit(roberta_model, dm_roberta)
        trainer_distil.fit(distil_model, dm_distil)

        bert_prob = get_probs(bert_model, dm_bert.test_dataloader())
        roberta_prob = get_probs(roberta_model, dm_roberta.test_dataloader())
        distil_prob = get_probs(distil_model, dm_distil.test_dataloader())

        labels = []
        with torch.no_grad():
            for batch in dm_roberta.test_dataloader():
                labels.append(batch["label"].cpu().numpy())
        y_true = np.concatenate(labels)

        bert_pred = (bert_prob >= 0.5).astype(int)
        roberta_pred = (roberta_prob >= 0.5).astype(int)
        distil_pred = (distil_prob >= 0.5).astype(int)

        votes = bert_pred + roberta_pred + distil_pred
        final_majority = (votes >= 2).astype(int)
        final_unanimous = (votes == 3).astype(int)

        print("=== MAJORITY VOTING ===")
        print("F1:", metrics.f1_score(y_true, final_majority))
        print("Precision:", metrics.precision_score(y_true, final_majority))
        print("Recall:", metrics.recall_score(y_true, final_majority))
        print("Accuracy:", metrics.accuracy_score(y_true, final_majority))

        print("\n=== UNANIMOUS VOTING ===")
        print("F1:", metrics.f1_score(y_true, final_unanimous))
        print("Precision:", metrics.precision_score(y_true, final_unanimous))
        print("Recall:", metrics.recall_score(y_true, final_unanimous))
        print("Accuracy:", metrics.accuracy_score(y_true, final_unanimous))

        all_true.extend(y_true)
        all_pred_majority.extend(final_majority)
        all_pred_unanimous.extend(final_unanimous)

        del bert_model, roberta_model, distil_model, dm_bert, dm_roberta, dm_distil, labels
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== FINAL RESULTS =====")
    print("\n===== MAJORITY VOTING ====")
    print("F1:", metrics.f1_score(all_true, all_pred_majority))
    print("Precision:", metrics.precision_score(all_true, all_pred_majority))
    print("Recall:", metrics.recall_score(all_true, all_pred_majority))
    print("Accuracy:", metrics.accuracy_score(all_true, all_pred_majority))

    print("\n=== UNANIMOUS VOTING ===")
    print("F1:", metrics.f1_score(all_true, all_pred_unanimous))
    print("Precision:", metrics.precision_score(all_true, all_pred_unanimous))
    print("Recall:", metrics.recall_score(all_true, all_pred_unanimous))
    print("Accuracy:", metrics.accuracy_score(all_true, all_pred_unanimous))

if __name__ == "__main__":
    main()