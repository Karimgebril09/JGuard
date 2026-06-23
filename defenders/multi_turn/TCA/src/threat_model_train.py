import numpy as np
import pandas as pd
import torch
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


def load_data(jigsaw_path: str, bias_path: str) -> pd.DataFrame:
    """load data and rename tables check null values and drop duplicates"""
    df_jigsaw = pd.read_csv(jigsaw_path)
    df_jigsaw = df_jigsaw[["comment_text", "threat"]].copy()
    df_jigsaw.columns = ["text", "label"]

    df_bias = pd.read_csv(bias_path)
    df_bias = df_bias[["comment_text", "threat"]].copy()
    df_bias.columns = ["text", "threat_score"]
    df_bias = df_bias.dropna(subset=["text", "threat_score"]).reset_index(drop=True)
    df_bias["label"] = (df_bias["threat_score"] >= 0.5).astype(int)

    threat_pos = df_bias[df_bias["label"] == 1]
    threat_neg = df_bias[df_bias["label"] == 0].sample(n=50000, random_state=42)
    df_bias = pd.concat([threat_pos, threat_neg]).sample(frac=1, random_state=42)
    df_bias = df_bias[["text", "label"]]

    df_all = pd.concat([df_jigsaw, df_bias], ignore_index=True)
    df_all = df_all.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df_all


def split_data(df: pd.DataFrame):
    """split data """
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=42
    )
    return train_df, val_df, test_df


def main():

    df = load_data(
        "defenders/multi_turn/integrated/data/threat/train.csv",
        "defenders/multi_turn/integrated/data/threat/all_data.csv",
    )

    train_df, _, _ = split_data(df)

    embedder = SentenceTransformer("all-mpnet-base-v2", device='cuda' if torch.cuda.is_available() else 'cpu')

    def embed(texts):
        return embedder.encode(list(texts), batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    X_train = embed(train_df["text"])

    y_train = train_df["label"].values

    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    joblib.dump(clf, "defenders/multi_turn/models/integrated/threat_classifier_lr1.joblib")


if __name__ == "__main__":
    main()