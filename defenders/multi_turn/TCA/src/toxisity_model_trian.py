import pandas as pd
import torch
import joblib
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

def load_data(path) :
    """load data and rename tables check null values"""
    df =pd.read_parquet(path)
    df_clean =df[["text", "label"]].copy()
    df_clean["label"] =pd.to_numeric(df_clean["label"], errors="coerce")
    df_clean =df_clean.dropna(subset=["text", "label"]).reset_index(drop=True)
    df_clean["label"] =df_clean["label"].astype(int)
    return df_clean

def split_data(df):
    """split data """
    train_df, test_df =train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    train_df, val_df =train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)
    return train_df, val_df, test_df

def main():
    df =load_data("defenders/multi_turn/integrated/data/toxisity/dynahate.parquet")
    train_df, _, _ =split_data(df)
    embedder =SentenceTransformer("all-mpnet-base-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
  
    X_train =embedder.encode(list(train_df["text"]), batch_size=64, show_progress_bar=True, convert_to_numpy=True )
    y_train =train_df["label"].values
    param ={"n_estimators": 594,"learning_rate": 0.166,"num_leaves": 128,"max_depth": 12,"subsample": 0.6,"colsample_bytree": 0.9965,}

    clf =lgb.LGBMClassifier(**param)
    clf.fit(X_train, y_train)

    joblib.dump(clf, "defenders/multi_turn/integrated/models/toxicity_lgb.joblib")

if __name__ =="__main__":
    main()