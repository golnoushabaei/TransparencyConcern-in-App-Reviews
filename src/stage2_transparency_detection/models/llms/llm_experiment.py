from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import Config
from src.stage2_transparency_detection.prompts.prompt_loader import PromptLoader
from src.stage2_transparency_detection.backends.azure_backend import AzureBackend
from src.stage2_transparency_detection.backends.openai_backend import OpenAIBackend
from src.stage2_transparency_detection.backends.bedrock_backend import BedrockBackend


def get_backend(cfg: Config):
    backend_name = cfg.run.backend.lower()
    max_tokens = cfg.run.max_completion_tokens

    if backend_name == "azure":
        return AzureBackend(cfg.azure, max_completion_tokens=max_tokens)
    elif backend_name == "openai":
        return OpenAIBackend(cfg.openai, max_completion_tokens=max_tokens)
    elif backend_name == "bedrock":
        return BedrockBackend(cfg.bedrock, max_completion_tokens=max_tokens)
    else:
        raise ValueError(
            f"Unsupported backend '{cfg.run.backend}'. "
            "Use 'azure', 'bedrock', or 'openai'."
        )


def parse_output(raw: str | None) -> tuple[int | None, str | None]:
    """Parse '<label>|<justification>' into (int|None, str|None)."""
    if not raw:
        return None, None
    text = raw.strip()
    if "|" not in text:
        return None, None
    label_part, justification = text.split("|", 1)
    label_part = label_part.strip()
    if label_part not in {"0", "1"}:
        return None, None
    return int(label_part), justification.strip()


def run_experiment(cfg: Config):
    files = cfg.files
    run_conf = cfg.run

    # loading prompt template from prompts/<name>.txt
    loader = PromptLoader()
    prompt_template = loader.load_prompt(cfg.prompt.template_name)

    # Backend (Azure / Bedrock / OpenAI)
    backend = get_backend(cfg)

    # loading data
    df = pd.read_csv(files.input_path)
    if files.text_col not in df.columns:
        raise ValueError(
            f"Missing column '{files.text_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    preds, justifications = [], []

    for review in tqdm(
        df[files.text_col].fillna("").astype(str),
        total=len(df),
        desc="Classifying",
    ):
        prompt = prompt_template.format(review=review)

        try:
            raw = backend.classify(prompt)
            pred, just = parse_output(raw)
        except Exception as e:
            print("Error during classification:", e)
            pred, just = None, None

        preds.append(pred)
        justifications.append(just)
        time.sleep(run_conf.sleep_seconds)

    df[files.pred_col] = preds
    df[files.just_col] = justifications

    files.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(files.output_path, index=False)
    print(f"Saved predictions to: {files.output_path}")

    # Optional metrics
    if files.ground_truth_col and files.ground_truth_col in df.columns:
        valid = df[files.pred_col].notna()
        y_true = df.loc[valid, files.ground_truth_col].astype(int)
        y_pred = df.loc[valid, files.pred_col].astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = None

        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        if auc is not None:
            print("ROC AUC:", auc)
        else:
            print("ROC AUC: could not be computed (single-class predictions).")
    else:
        print("No ground truth provided, so accuracy/F1 cannot be computed.")


def main():
    cfg = Config.get_default_config()
    run_experiment(cfg)


if __name__ == "__main__":
    main()