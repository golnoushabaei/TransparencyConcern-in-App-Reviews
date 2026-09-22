from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from collections import defaultdict
from umap import UMAP
from hdbscan import HDBSCAN

from config import Config
from src.stage2_transparency_detection.prompts.prompt_loader import PromptLoader
from src.stage2_transparency_detection.backends.azure_backend import AzureBackend
from src.stage2_transparency_detection.backends.openai_backend import OpenAIBackend
from src.stage2_transparency_detection.backends.bedrock_backend import BedrockBackend


# ========= CONFIG / PATHS =========

GT_PATH = Path("data/ground_truth/ground-truth-final.csv")  # label 1 subset precomputed
FINAL_PREDICTIONS_PATH = Path("results/stage3/final_predictions.csv")     # from RoBERTa script
APPS_PATH = Path("data/apps/Apps.xlsx")
OUTPUT_TOPICS_CSV = Path("results/stage3/topics_experiments.csv")

# If you pre-saved label-1 subsets, you can read them directly:
GT_LABEL1_PATH = Path("data/ground_truth/label1_ground_truth.csv")
FINAL_PRED_LABEL1_PATH = Path("results/stage3/label1_final_predictions.csv")


# ========= LLM BACKEND SELECTION =========

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


# ========= DATA PREP =========

def load_positives():
    """
    Load label=1 rows from:
      - ground truth (Label == 1)
      - RoBERTa final predictions (pred_label == 1)

    If the precomputed CSVs exist, use them.
    Otherwise, generate them from the full files.
    """
    # Paths
    gt_full = Path("data/ground_truth/ground-truth-final.csv")
    preds_full = FINAL_PREDICTIONS_PATH  # results/stage3/final_predictions.csv

    gt_label1_path = GT_LABEL1_PATH
    pred_label1_path = FINAL_PRED_LABEL1_PATH

    # Ground-truth label 1
    if gt_label1_path.exists():
        gt_positives_df = pd.read_csv(gt_label1_path)
        print(f"Loaded GT positives from {gt_label1_path} ({len(gt_positives_df)} rows)")
    else:
        gt_df = pd.read_csv(gt_full)
        gt_positives_df = gt_df[gt_df["Label"] == 1].copy()
        gt_label1_path.parent.mkdir(parents=True, exist_ok=True)
        gt_positives_df.to_csv(gt_label1_path, index=False)
        print(f"Computed GT positives ({len(gt_positives_df)} rows) and saved to {gt_label1_path}")

    # RoBERTa predictions label 1
    if pred_label1_path.exists():
        final_positives_df = pd.read_csv(pred_label1_path)
        print(f"Loaded RoBERTa positives from {pred_label1_path} ({len(final_positives_df)} rows)")
    else:
        preds_df = pd.read_csv(preds_full)
        final_positives_df = preds_df[preds_df["pred_label"] == 1].copy()
        pred_label1_path.parent.mkdir(parents=True, exist_ok=True)
        final_positives_df.to_csv(pred_label1_path, index=False)
        print(f"Computed RoBERTa positives ({len(final_positives_df)} rows) and saved to {pred_label1_path}")

    positives_df = pd.concat([final_positives_df, gt_positives_df], ignore_index=True)
    print("Number of rows (positives):", len(positives_df))
    return positives_df


def build_vectorizer(app_names):
    stopwords = set(ENGLISH_STOP_WORDS)
    # keeping 'no' if you want
    for w in ["no"]:
        stopwords.discard(w)
    stopwords.update(app_names)
    stopwords = list(stopwords)

    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=5,  # ignore terms that appear in <5 reviews
    )
    return vectorizer_model


def compute_silhouette(embeddings, topics):
    topics = np.array(topics)
    mask = topics != -1
    if len(set(topics[mask])) < 2:
        return -1
    return silhouette_score(embeddings[mask], topics[mask])


# ========= BERTopic pipeline =========

def run_bertopic(positives_df: pd.DataFrame):
    # App names -> stopwords
    apps_df = pd.read_excel(APPS_PATH)
    app_names = apps_df["AppName"].astype(str).str.lower().str.strip().tolist()

    texts = positives_df["clean_review"].astype(str).tolist()

    vectorizer_model = build_vectorizer(app_names)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode(texts)

    base_model = BERTopic(
        embedding_model=embed_model,
        language="english",
        vectorizer_model=vectorizer_model,
        min_topic_size=100,
        n_gram_range=(1, 2),
        verbose=True,
    )

    topics, probs = base_model.fit_transform(texts, embeddings)
    print("Unique topics (excluding -1/noise):", len({t for t in topics if t != -1}))

    original_model = base_model

    # Explore different nr_topics with silhouette
    nr_topics_list = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
    results = []
    for k in nr_topics_list:
        model_copy = copy.deepcopy(original_model)
        reduced_model = model_copy.reduce_topics(texts, nr_topics=k)
        reduced_topics = reduced_model.topics_
        sil = compute_silhouette(embeddings, reduced_topics)
        print(f"nr_topics={k} | Silhouette={sil:.4f}")
        results.append({"nr_topics": k, "silhouette": sil})
    results_df = pd.DataFrame(results)
    print(results_df)

    # UMAP + HDBSCAN tuning
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    topic_model_2 = BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=50,
        verbose=True,
    )

    topics_2, probs_2 = topic_model_2.fit_transform(texts, embeddings)
    print("Unique topics (excluding -1/noise):", len({t for t in topics_2 if t != -1}))

    model_2 = topic_model_2
    nr_topics_list_2 = [5, 6, 7, 10, 20, 30, 40, 45]
    results_2 = []
    for k in nr_topics_list_2:
        model_copy_2 = copy.deepcopy(model_2)
        reduced_model_2 = model_copy_2.reduce_topics(texts, nr_topics=k)
        reduced_topics_2 = reduced_model_2.topics_
        sil = compute_silhouette(embeddings, reduced_topics_2)
        print(f"nr_topics={k} | Silhouette={sil:.4f}")
        results_2.append({"nr_topics": k, "silhouette": sil})
    results_2_df = pd.DataFrame(results_2)
    print(results_2_df)

    model_copy_2 = copy.deepcopy(model_2)
    reduced_model_2_45 = model_copy_2.reduce_topics(texts, nr_topics=45)
    reduced_topics_2_45 = reduced_model_2_45.topics_
    positives_df["reduced_topics_2_45"] = reduced_topics_2_45

    return positives_df, reduced_model_2_45, texts


# ========= LLM-based topic reduction =========

def extract_topic_keywords(topic_model, top_n_words=10, include_outliers=False):
    topic_info = topic_model.get_topic_info()
    topic_rows = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1 and not include_outliers:
            continue
        topic_words = topic_model.get_topic(topic_id)
        if topic_words is None:
            continue
        keywords = [word for word, score in topic_words[:top_n_words]]
        keyword_scores = [score for word, score in topic_words[:top_n_words]]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "keywords": keywords,
                "keyword_scores": keyword_scores,
            }
        )
    return pd.DataFrame(topic_rows)


def topics_to_prompt_text(topic_keywords_df):
    lines = []
    for _, row in topic_keywords_df.iterrows():
        topic_id = row["topic_id"]
        keywords = ", ".join(row["keywords"])
        lines.append(f"Topic {topic_id}: {keywords}")
    return "\n".join(lines)


def extract_topics_with_examples(topic_model, texts, top_n_words=10, n_examples=3, include_outliers=False):
    topic_info = topic_model.get_topic_info()
    topic_assignments = topic_model.topics_

    topic_to_docs = {}
    for doc, topic in zip(texts, topic_assignments):
        topic_to_docs.setdefault(topic, []).append(doc)

    topic_rows = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1 and not include_outliers:
            continue
        topic_words = topic_model.get_topic(topic_id)
        if topic_words is None:
            continue
        keywords = [word for word, score in topic_words[:top_n_words]]
        examples = topic_to_docs.get(topic_id, [])[:n_examples]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "keywords": keywords,
                "examples": examples,
            }
        )
    return pd.DataFrame(topic_rows)


def topics_with_examples_to_prompt_text(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(f"Topic {row['topic_id']}")
        lines.append(f"Keywords: {', '.join(row['keywords'])}")
        for i, ex in enumerate(row["examples"], 1):
            lines.append(f"Example {i}: {ex[:400]}")
        lines.append("")
    return "\n".join(lines)


def run_llm_reduction(cfg: Config, topic_model, texts, positives_df):
    loader = PromptLoader()
    # Your prompt file should have a `{topics}` placeholder
    prompt_template = loader.load_prompt("bertopic-reduction-prompt")

    backend = get_backend(cfg)

    # ========= 1) First LLM reduction: keywords only =========
    topic_keywords_df = extract_topic_keywords(topic_model, top_n_words=10)
    topics_text = topics_to_prompt_text(topic_keywords_df)

    llm_prompt_1 = prompt_template.format(topics=topics_text)
    print("\n=== LLM Reduction #1 (keywords only) ===")
    response_1 = backend.classify(llm_prompt_1)
    llm_merges_1 = json.loads(response_1)

    topic_to_final_theme_1 = {}
    for item in llm_merges_1:
        label = item["final_theme_label"]
        for topic_id in item["merged_topic_ids"]:
            topic_to_final_theme_1[topic_id] = label

    original_topics = topic_model.topics_
    final_theme_assignments_1 = []
    for topic in original_topics:
        if topic == -1:
            final_theme_assignments_1.append("Outlier")
        else:
            final_theme_assignments_1.append(
                topic_to_final_theme_1.get(topic, f"Unmapped_{topic}")
            )

    df_final_1 = pd.DataFrame(
        {
            "text": texts,
            "reduced_topic_id": topic_model.topics_,
            "final_theme": final_theme_assignments_1,
        }
    )
    print("\nTheme counts after LLM reduction #1:")
    print(df_final_1["final_theme"].value_counts())

    # ========= 2) Second LLM reduction: keywords + examples =========
    topic_examples_df = extract_topics_with_examples(
        topic_model,
        texts,
        top_n_words=10,
        n_examples=3,
    )
    topics_text_with_examples = topics_with_examples_to_prompt_text(topic_examples_df)

    llm_prompt_2 = prompt_template.format(topics=topics_text_with_examples)
    print("\n=== LLM Reduction #2 (keywords + examples) ===")
    response_2 = backend.classify(llm_prompt_2)
    llm_merges_2 = json.loads(response_2)

    topic_to_final_theme_2 = {}
    for item_2 in llm_merges_2:
        label_2 = item_2["final_theme_label"]
        for topic_id_2 in item_2["merged_topic_ids"]:
            topic_to_final_theme_2[topic_id_2] = label_2

    final_theme_assignments_2 = []
    for topic_2 in original_topics:
        if topic_2 == -1:
            final_theme_assignments_2.append("Outlier")
        else:
            final_theme_assignments_2.append(
                topic_to_final_theme_2.get(topic_2, f"Unmapped_{topic_2}")
            )

    df_final_2 = pd.DataFrame(
        {
            "text": texts,
            "reduced_topic_id": topic_model.topics_,
            "final_theme": final_theme_assignments_2,
        }
    )
    print("\nTheme counts after LLM reduction #2:")
    print(df_final_2["final_theme"].value_counts())

    # ========= Build final mappings (from second reduction) =========

    llm_theme_to_id_2 = {}
    llm_topic_id_to_label_2 = {}
    original_topic_to_llm_id_2 = {}
    original_topic_to_llm_label_2 = {}

    for new_id_2, item_2 in enumerate(llm_merges_2):
        label_2 = item_2["final_theme_label"]
        llm_theme_to_id_2[label_2] = new_id_2
        llm_topic_id_to_label_2[new_id_2] = label_2
        for old_topic_id_2 in item_2["merged_topic_ids"]:
            original_topic_to_llm_id_2[old_topic_id_2] = new_id_2
            original_topic_to_llm_label_2[old_topic_id_2] = label_2

    all_reduced_topics = sorted(set(t for t in topic_model.topics_ if t != -1))
    mapped_topics = set(original_topic_to_llm_id_2.keys())
    unmapped_topics = sorted(set(all_reduced_topics) - mapped_topics)
    print("Unmapped topics after LLM reduction #2:", unmapped_topics)

    # assigning new IDs to unmapped topics
    next_llm_id_2 = max(llm_topic_id_to_label_2.keys(), default=-1) + 1
    for topic_id in unmapped_topics:
        new_label = f"Unmapped_{topic_id}"
        original_topic_to_llm_id_2[topic_id] = next_llm_id_2
        original_topic_to_llm_label_2[topic_id] = new_label
        llm_topic_id_to_label_2[next_llm_id_2] = new_label
        next_llm_id_2 += 1

    positives_df_copy = positives_df.copy()
    positives_df_copy["llm_topic_id_2"] = positives_df_copy["reduced_topics_2_45"].apply(
        lambda t: -1 if t == -1 else original_topic_to_llm_id_2.get(t, None)
    )
    positives_df_copy["llm_topic_label_2"] = positives_df_copy["reduced_topics_2_45"].apply(
        lambda t: "Outlier" if t == -1 else original_topic_to_llm_label_2.get(t, f"Unmapped_{t}")
    )

    OUTPUT_TOPICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    positives_df_copy.to_csv(OUTPUT_TOPICS_CSV, index=False)
    print(f"\nTopics with LLM labels (second reduction) saved to {OUTPUT_TOPICS_CSV}")


def main():
    cfg = Config.get_default_config()

    # 1. Load positives (label=1 GT + label=1 RoBERTa)
    positives_df = load_positives()

    # 2. Run BERTopic + reduction
    positives_df, reduced_model_2_45, texts = run_bertopic(positives_df)

    # 3. LLM-based topic reduction
    run_llm_reduction(cfg, reduced_model_2_45, texts, positives_df)


if __name__ == "__main__":
    main()