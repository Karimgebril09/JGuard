from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


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


def _tokenize_dataset(
    dataset: Dataset,
    *,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }


def _build_training_args(
    *,
    output_dir: Path,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    logging_steps: int,
    use_fp16: bool,
) -> TrainingArguments:
    return TrainingArguments(  # pyright: ignore[reportCallIssue]
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=logging_steps,
        fp16=use_fp16,
        report_to="none",
    )


def train_and_evaluate(
    *,
    final_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    test_size: float,
    random_state: int,
    max_length: int,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    logging_steps: int,
) -> dict[str, Any]:
    working = final_df.copy()
    working["binary_target"] = working["mapped_category"].map(
        lambda category: "safe" if category == SAFE_LABEL else "unsafe"
    )

    (
        X_train,
        X_test,
        y_binary_train_text,
        y_binary_test_text,
        y_category_train_text,
        y_category_test_text,
    ) = train_test_split(
        working["prompt"].tolist(),
        working["binary_target"].tolist(),
        working["mapped_category"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=working["binary_target"].tolist(),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_fp16 = bool(torch.cuda.is_available())

    binary_encoder = LabelEncoder()
    y_binary_train = binary_encoder.fit_transform(y_binary_train_text)
    y_binary_test = binary_encoder.transform(y_binary_test_text)

    train_binary_ds = Dataset.from_dict(
        {"text": X_train, "label": np.asarray(y_binary_train).tolist()}
    )
    test_binary_ds = Dataset.from_dict(
        {"text": X_test, "label": np.asarray(y_binary_test).tolist()}
    )
    train_binary_ds = _tokenize_dataset(train_binary_ds, tokenizer=tokenizer, max_length=max_length)
    test_binary_ds = _tokenize_dataset(test_binary_ds, tokenizer=tokenizer, max_length=max_length)

    binary_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(binary_encoder.classes_),
    )

    binary_args = _build_training_args(
        output_dir=output_dir / "binary_run",
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        use_fp16=use_fp16,
    )

    binary_trainer = Trainer(
        model=binary_model,
        args=binary_args,
        train_dataset=train_binary_ds,
        eval_dataset=test_binary_ds,
        compute_metrics=_compute_metrics,
    )
    binary_trainer.train()

    binary_predictions = binary_trainer.predict(cast(Any, test_binary_ds))
    y_binary_pred = np.argmax(binary_predictions.predictions, axis=1)
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

    unsafe_train_rows = [i for i, label in enumerate(y_binary_train_text) if label == "unsafe"]
    unsafe_test_rows = [i for i, label in enumerate(y_binary_test_text) if label == "unsafe"]
    X_category_train = [X_train[i] for i in unsafe_train_rows]
    X_category_test = [X_test[i] for i in unsafe_test_rows]
    y_category_train_filtered = [y_category_train_text[i] for i in unsafe_train_rows]
    y_category_test_filtered = [y_category_test_text[i] for i in unsafe_test_rows]

    if not X_category_train:
        raise RuntimeError("No unsafe training samples available for category fine-tuning.")

    category_encoder = LabelEncoder()
    y_category_train = category_encoder.fit_transform(y_category_train_filtered)

    train_category_ds = Dataset.from_dict(
        {"text": X_category_train, "label": np.asarray(y_category_train).tolist()}
    )
    train_category_ds = _tokenize_dataset(
        train_category_ds, tokenizer=tokenizer, max_length=max_length
    )

    test_category_ds: Dataset | None = None
    y_category_test_np: Any = None
    if X_category_test:
        y_category_test_np = np.asarray(
            category_encoder.transform(y_category_test_filtered)
        ).tolist()
        test_category_ds = Dataset.from_dict(
            {"text": X_category_test, "label": y_category_test_np}
        )
        test_category_ds = _tokenize_dataset(
            test_category_ds, tokenizer=tokenizer, max_length=max_length
        )

    category_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(category_encoder.classes_),
    )

    category_args = _build_training_args(
        output_dir=output_dir / "category_run",
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        use_fp16=use_fp16,
    )
    category_args.eval_strategy = "no" if test_category_ds is None else "epoch"
    category_args.save_strategy = "epoch"

    category_trainer = Trainer(
        model=category_model,
        args=category_args,
        train_dataset=train_category_ds,
        eval_dataset=test_category_ds,
        compute_metrics=_compute_metrics,
    )
    category_trainer.train()

    category_report_text = "No unsafe samples in test split; category report unavailable."
    category_report_json: dict[str, Any] = {}
    if test_category_ds is not None and y_category_test_np is not None:
        category_predictions = category_trainer.predict(cast(Any, test_category_ds))
        y_category_pred = np.argmax(category_predictions.predictions, axis=1)
        category_report_text = cast(
            str,
            classification_report(
                y_category_test_np,
                y_category_pred,
                target_names=category_encoder.classes_,
            ),
        )
        category_report_json = cast(
            dict[str, Any],
            classification_report(
                y_category_test_np,
                y_category_pred,
                target_names=category_encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
        )

    final_binary_dir = output_dir / "binary_final"
    final_category_dir = output_dir / "unsafe_category_final"
    final_binary_dir.mkdir(parents=True, exist_ok=True)
    final_category_dir.mkdir(parents=True, exist_ok=True)

    binary_trainer.model.save_pretrained(final_binary_dir)
    category_trainer.model.save_pretrained(final_category_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    joblib.dump(binary_encoder, output_dir / "binary_label_encoder.pkl")
    joblib.dump(category_encoder, output_dir / "unsafe_category_label_encoder.pkl")

    metadata = {
        "model_name": model_name,
        "num_samples": len(final_df),
        "binary_label_distribution": {
            "safe": int((final_df["mapped_category"] == SAFE_LABEL).sum()),
            "unsafe": int((final_df["mapped_category"] != SAFE_LABEL).sum()),
        },
        "category_label_distribution": final_df["mapped_category"].value_counts().to_dict(),
        "binary_labels": list(binary_encoder.classes_),
        "category_labels": list(category_encoder.classes_),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_unsafe_size": len(X_category_test),
        "epochs": epochs,
        "max_length": max_length,
        "use_fp16": use_fp16,
        "random_state": random_state,
    }

    report_text = (
        "Primary binary evaluation (safe vs unsafe):\n"
        + binary_report_text
        + "\nAuxiliary unsafe-category evaluation:\n"
        + category_report_text
    )
    report_json = {
        "binary": binary_report_json,
        "unsafe_category": category_report_json,
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    return {
        "binary_report_text": binary_report_text,
        "category_report_text": category_report_text,
        "report_text": report_text,
        "report_json": report_json,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for stage-8 binary safe/unsafe plus unsafe-category prediction."
    )
    parser.add_argument("--split", default="30k_train", help="BeaverTails split to use.")
    parser.add_argument("--safe-samples", type=int, default=2500)
    parser.add_argument("--max-unsafe-per-class", type=int, default=2500)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "distilbert_artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_df, source_prompt_column = prepare_dataset(
        split=args.split,
        safe_samples=args.safe_samples,
        max_unsafe_per_class=args.max_unsafe_per_class,
        max_total_rows=args.max_total_rows,
        random_state=args.random_state,
    )
    if final_df.empty:
        raise RuntimeError("No training samples produced after remapping and filtering.")

    training_result = train_and_evaluate(
        final_df=final_df,
        output_dir=args.output_dir,
        model_name=args.model_name,
        test_size=args.test_size,
        random_state=args.random_state,
        max_length=args.max_length,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
    )

    metadata = dict(training_result["metadata"])
    metadata["source_prompt_column"] = source_prompt_column
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Category label distribution:")
    print(final_df["mapped_category"].value_counts())
    print("\nBinary label distribution:")
    print(metadata["binary_label_distribution"])
    print("\nPrimary binary evaluation (safe vs unsafe):")
    print(training_result["binary_report_text"])
    print("\nAuxiliary unsafe-category evaluation:")
    print(training_result["category_report_text"])
    print(f"\nSaved artifacts to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
