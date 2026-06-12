from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from datasets import load_dataset
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CATEGORY_MAP: dict[str, str] = {
    "privacy_violation": "data_exposure",
    "financial_crime,property_crime,theft": "illegal_actions",
    "drug_abuse,weapons,banned_substance": "illegal_actions",
    "terrorism,organized_crime": "illegal_actions",
    "violence,aiding_and_abetting,incitement": "illegal_actions",
    "non_violent_unethical_behavior": "illegal_actions",
    "hate_speech,offensive_language": "toxicity",
    "discrimination,stereotype,injustice": "toxicity",
    "sexually_explicit,adult_content": "toxicity",
    "self_harm": "toxicity",
    "child_abuse": "toxicity",
    "animal_abuse": "toxicity",
    "misinformation_regarding_ethics,laws_and_safety": "hallucination_facilitation",
    "controversial_topics,politics": "hallucination_facilitation",
}

CATEGORY_PRIORITY: tuple[str, ...] = (
    "data_exposure",
    "illegal_actions",
    "toxicity",
    "hallucination_facilitation",
)

SAFE_LABEL = "safe"


def _resolve_prompt_column(df: pd.DataFrame) -> str:
    candidate_columns = ("prompt", "instruction", "question")
    for column in candidate_columns:
        if column in df.columns:
            return column
    raise ValueError(
        "Unable to find prompt text column. Expected one of "
        f"{candidate_columns}, got {list(df.columns)}"
    )


def _extract_active_categories(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key, enabled in value.items() if bool(enabled)]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str) and item]
    if isinstance(value, str) and value:
        return [value]
    return []


def _pick_mapped_category(active_categories: list[str]) -> str | None:
    mapped = [CATEGORY_MAP[c] for c in active_categories if c in CATEGORY_MAP]
    if not mapped:
        return None

    for preferred in CATEGORY_PRIORITY:
        if preferred in mapped:
            return preferred
    return mapped[0]


def _sample_grouped(
    df: pd.DataFrame,
    *,
    label_column: str,
    max_per_label: int,
    random_state: int,
) -> pd.DataFrame:
    sampled = []
    for _, group in df.groupby(label_column, sort=False):
        sampled.append(
            group.sample(n=min(max_per_label, len(group)), random_state=random_state)
        )
    return pd.concat(sampled, ignore_index=True)


def prepare_dataset(
    *,
    split: str,
    safe_samples: int,
    max_unsafe_per_class: int | None,
    max_total_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, str]:
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    df = cast(pd.DataFrame, ds.to_pandas())
    prompt_column = _resolve_prompt_column(df)

    working = df.copy()
    working["active_categories"] = working["category"].map(_extract_active_categories)
    working["mapped_category"] = working["active_categories"].map(_pick_mapped_category)

    unsafe_df = working[(working["is_safe"] == False) & (working["mapped_category"].notna())]
    if max_unsafe_per_class is not None:
        unsafe_df = _sample_grouped(
            unsafe_df,
            label_column="mapped_category",
            max_per_label=max_unsafe_per_class,
            random_state=random_state,
        )

    safe_df = working[working["is_safe"] == True]
    safe_df = safe_df.sample(n=min(safe_samples, len(safe_df)), random_state=random_state).copy()
    safe_df["mapped_category"] = SAFE_LABEL

    final_df = pd.concat([unsafe_df, safe_df], ignore_index=True)
    final_df = final_df[[prompt_column, "mapped_category"]].rename(
        columns={prompt_column: "prompt"}
    )
    final_df = final_df.dropna(subset=["prompt", "mapped_category"]).reset_index(drop=True)

    if max_total_rows is not None and max_total_rows < len(final_df):
        _, final_df = train_test_split(
            final_df,
            test_size=max_total_rows,
            random_state=random_state,
            stratify=final_df["mapped_category"],
        )
        final_df = final_df.reset_index(drop=True)

    final_df = final_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return final_df, prompt_column


def train_baseline(
    *,
    final_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    max_features: int,
) -> dict[str, Any]:
    modeling_df = final_df.copy()
    modeling_df["binary_target"] = modeling_df["mapped_category"].map(
        lambda category: "safe" if category == SAFE_LABEL else "unsafe"
    )

    X = modeling_df["prompt"].tolist()
    y_binary_text = modeling_df["binary_target"].tolist()
    y_category_text = modeling_df["mapped_category"].tolist()

    (
        X_train,
        X_test,
        y_binary_train_text,
        y_binary_test_text,
        y_category_train_text_all,
        y_category_test_text_all,
    ) = train_test_split(
        X,
        y_binary_text,
        y_category_text,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary_text,
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    binary_encoder = LabelEncoder()
    y_binary_train = binary_encoder.fit_transform(y_binary_train_text)
    y_binary_test = binary_encoder.transform(y_binary_test_text)

    binary_classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    binary_classifier.fit(X_train_tfidf, y_binary_train)

    y_binary_pred = binary_classifier.predict(X_test_tfidf)
    binary_report_text = cast(
        str,
        classification_report(
            y_binary_test,
            y_binary_pred,
            target_names=binary_encoder.classes_,
        ),
    )
    binary_report_json = cast(
        dict[str, Any],
        classification_report(
            y_binary_test,
            y_binary_pred,
            target_names=binary_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    )

    unsafe_train_df = modeling_df[modeling_df["mapped_category"] != SAFE_LABEL]
    category_encoder = LabelEncoder()
    category_encoder.fit(unsafe_train_df["mapped_category"])

    category_classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    X_train_tfidf_csr = csr_matrix(X_train_tfidf)
    X_test_tfidf_csr = csr_matrix(X_test_tfidf)

    train_unsafe_indices = [
        index for index, label in enumerate(y_binary_train_text) if label == "unsafe"
    ]
    X_category_train = X_train_tfidf_csr[train_unsafe_indices]
    y_category_train_text = [
        category
        for category, binary_label in zip(
            y_category_train_text_all, y_binary_train_text, strict=False
        )
        if binary_label == "unsafe"
    ]

    if len(y_category_train_text) == 0:
        raise RuntimeError("No unsafe samples found for category classifier training.")

    y_category_train = category_encoder.transform(y_category_train_text)
    category_classifier.fit(X_category_train, y_category_train)

    test_unsafe_indices = [
        index for index, label in enumerate(y_binary_test_text) if label == "unsafe"
    ]
    category_report_text = "No unsafe samples in test split; category report unavailable."
    category_report_json: dict[str, Any] = {}
    if test_unsafe_indices:
        X_category_test = X_test_tfidf_csr[test_unsafe_indices]
        y_category_test_text = [y_category_test_text_all[idx] for idx in test_unsafe_indices]
        y_category_test = category_encoder.transform(y_category_test_text)
        y_category_pred = category_classifier.predict(X_category_test)
        category_report_text = cast(
            str,
            classification_report(
                y_category_test,
                y_category_pred,
                target_names=category_encoder.classes_,
            ),
        )
        category_report_json = cast(
            dict[str, Any],
            classification_report(
                y_category_test,
                y_category_pred,
                target_names=category_encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
        )

    report_text = (
        "Binary task (primary): safe vs unsafe\n"
        + binary_report_text
        + "\nAuxiliary unsafe-category task:\n"
        + category_report_text
    )
    report_json = {
        "binary": binary_report_json,
        "unsafe_category": category_report_json,
    }

    return {
        "binary_encoder": binary_encoder,
        "category_encoder": category_encoder,
        "vectorizer": vectorizer,
        "binary_classifier": binary_classifier,
        "category_classifier": category_classifier,
        "binary_report_text": binary_report_text,
        "binary_report_json": binary_report_json,
        "category_report_text": category_report_text,
        "category_report_json": category_report_json,
        "report_text": report_text,
        "report_json": report_json,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "binary_labels": list(binary_encoder.classes_),
        "category_labels": list(category_encoder.classes_),
    }


def save_artifacts(
    *,
    output_dir: Path,
    training_result: dict[str, Any],
    final_df: pd.DataFrame,
    split: str,
    source_prompt_column: str,
    random_state: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(training_result["vectorizer"], output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(training_result["binary_classifier"], output_dir / "binary_classifier.joblib")
    joblib.dump(training_result["binary_encoder"], output_dir / "binary_label_encoder.joblib")
    joblib.dump(training_result["category_classifier"], output_dir / "unsafe_category_classifier.joblib")
    joblib.dump(training_result["category_encoder"], output_dir / "unsafe_category_label_encoder.joblib")

    metadata = {
        "dataset": "PKU-Alignment/BeaverTails",
        "split": split,
        "source_prompt_column": source_prompt_column,
        "num_samples": len(final_df),
        "label_distribution": final_df["mapped_category"].value_counts().to_dict(),
        "binary_label_distribution": {
            "safe": int((final_df["mapped_category"] == SAFE_LABEL).sum()),
            "unsafe": int((final_df["mapped_category"] != SAFE_LABEL).sum()),
        },
        "train_size": training_result["train_size"],
        "test_size": training_result["test_size"],
        "binary_labels": training_result["binary_labels"],
        "category_labels": training_result["category_labels"],
        "random_state": random_state,
    }

    (output_dir / "report.txt").write_text(training_result["report_text"], encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(training_result["report_json"], indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + Logistic Regression stage-8 harm classifier on BeaverTails."
    )
    parser.add_argument("--split", default="30k_train", help="BeaverTails split to use.")
    parser.add_argument(
        "--safe-samples",
        type=int,
        default=2000,
        help="Number of safe samples to add to the training table.",
    )
    parser.add_argument(
        "--max-unsafe-per-class",
        type=int,
        default=None,
        help="Optional cap on unsafe samples per mapped class.",
    )
    parser.add_argument(
        "--max-total-rows",
        type=int,
        default=None,
        help="Optional cap on the final dataset size after balancing.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory where model artifacts and reports will be saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    final_df, source_prompt_column = prepare_dataset(
        split=args.split,
        safe_samples=args.safe_samples,
        max_unsafe_per_class=args.max_unsafe_per_class,
        max_total_rows=args.max_total_rows,
        random_state=args.random_state,
    )

    if final_df.empty:
        raise RuntimeError("No training samples produced after remapping and filtering.")

    training_result = train_baseline(
        final_df=final_df,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )

    save_artifacts(
        output_dir=args.output_dir,
        training_result=training_result,
        final_df=final_df,
        split=args.split,
        source_prompt_column=source_prompt_column,
        random_state=args.random_state,
    )

    print("Label distribution:")
    print(final_df["mapped_category"].value_counts())
    print("\nBinary label distribution:")
    print(
        {
            "safe": int((final_df["mapped_category"] == SAFE_LABEL).sum()),
            "unsafe": int((final_df["mapped_category"] != SAFE_LABEL).sum()),
        }
    )
    print("\nPrimary binary evaluation (safe vs unsafe):")
    print(training_result["binary_report_text"])
    print("\nAuxiliary unsafe-category evaluation:")
    print(training_result["category_report_text"])
    print(f"\nSaved artifacts to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
