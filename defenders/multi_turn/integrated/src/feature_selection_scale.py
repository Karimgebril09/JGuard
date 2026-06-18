import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV,VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PowerTransformer
from defenders.multi_turn.integrated.inference.transforms import TRANSFORMS

os.makedirs("data/processed",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("config",exist_ok=True)


def conversation_split(df):
    all_conv_ids = df["conv_id"].unique()
    train_val_ids, test_ids = train_test_split(all_conv_ids, test_size=0.2, random_state=42, shuffle=True, stratify=df.groupby("conv_id")["label"].max())
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.25, random_state=42, shuffle=True, stratify=df[df["conv_id"].isin(train_val_ids)].groupby("conv_id")["label"].max())

    train_df = df[df["conv_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["conv_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["conv_id"].isin(test_ids)].reset_index(drop=True)

    return {"train": train_df, "val": val_df, "test": test_df}


def remove_multicollinear(X_train,y_train,threshold):
    corr_df= X_train.copy()
    corr_df["target"]= y_train.values
    corr_matrix= corr_df.corr()

    upper= corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    to_remove= set()
    for col in upper.columns:
        if col== "target":
            continue
        highly_correlated= upper.index[abs(upper[col]) > threshold].tolist()
        for other_col in highly_correlated:
            if other_col== "target" or other_col in to_remove or col in to_remove:
                continue
            corr_col= abs(corr_matrix.loc[col,"target"])
            corr_other_col= abs(corr_matrix.loc[other_col,"target"])
            if corr_col >= corr_other_col: # will remove the one that less correlate with the 
                to_remove.add(other_col)
            else:
                to_remove.add(col)

    cols_to_keep= [c for c in X_train.columns if c not in to_remove]
    return cols_to_keep


def save_split(X,meta,y,path):
    out= X.copy()
    out["conv_id"]= meta["conv_id"].values
    out["turn_id"]= meta["turn_id"].values
    out["label"]= y.values
    out.to_csv(path,index=False)
    print(f"Saved {path}  shape={out.shape}")


def transform_feature(series, transform):
    if transform == "log1p":
        return np.log1p(np.maximum(series, 0))

    if transform == "square":
        return np.square(series)

    if transform == "binarize":
        return (series > 0).astype(float)
    if transform == "yeo-johnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        return pt.fit_transform(
            np.asarray(series).reshape(-1, 1)
        ).flatten()

    return series

def apply_transform(df) :
    df = df.copy()
    for feature, transform in TRANSFORMS.items():
        if feature not in df.columns:
            continue
        df[f"{feature}"] = transform_feature(df[feature], transform)
    return df


def main():
    # df = pd.read_csv("data/merged/merged_features.csv")

    df= pd.read_csv("data/total/features_before_selection(1).csv")
    data= conversation_split(df)
    train_df= data["train"]
    val_df= data["val"]
    test_df= data["test"]
    feature_cols = [c for c in train_df.columns if c not in ["conv_id", "turn_id", "label"]]
    meta_train= train_df[["conv_id","turn_id"]]
    meta_val= val_df[["conv_id","turn_id"]]
    meta_test= test_df[["conv_id","turn_id"]]

    X_train= train_df[feature_cols]
    X_val= val_df[feature_cols]
    X_test= test_df [feature_cols]

    y_train= train_df["label"]
    y_val= val_df["label"]
    y_test= test_df["label"]
  

    X_train= apply_transform(X_train)
    X_val= apply_transform(X_val)
    X_test= apply_transform(X_test)


    var_sel= VarianceThreshold(threshold=0.005)
    var_sel.fit(X_train)
    var_cols= X_train.columns[var_sel.get_support()].tolist()
    X_train_var= pd.DataFrame(var_sel.transform(X_train),columns=var_cols)
    X_val_var= pd.DataFrame(var_sel.transform(X_val),columns=var_cols)
    X_test_var= pd.DataFrame(var_sel.transform(X_test),columns=var_cols)
    print(f"After variance filter:{len(var_cols)}")


    corr_vals= X_train_var.corrwith(y_train,method="spearman").abs()
    corr_cols= corr_vals[corr_vals > 0.005].index.tolist()
    X_train_corr= X_train_var[corr_cols]
    X_val_corr= X_val_var[corr_cols]
    X_test_corr= X_test_var[corr_cols]
    print(f"After correlation filter:{len(corr_cols)}")

    mc_cols= remove_multicollinear(X_train_corr,y_train,0.9)


    X_train_final= X_train_corr[mc_cols]
    X_val_final= X_val_corr[mc_cols]
    X_test_final= X_test_corr[mc_cols]
    print(f"After multicollinearity:{len(mc_cols)}")


    final_cols = X_train_final.columns.tolist()
    scaler= RobustScaler()
    X_train_final= pd.DataFrame(scaler.fit_transform(X_train_final), columns=final_cols)
    X_val_final= pd.DataFrame(scaler.transform(X_val_final), columns=final_cols)
    X_test_final= pd.DataFrame(scaler.transform(X_test_final), columns=final_cols)
    joblib.dump(scaler,"models/scaler(1).pkl")

    save_split(X_train_final,meta_train,y_train,f"data/processed/train(1).csv")
    save_split(X_val_final,meta_val,y_val,f"data/processed/validation(1).csv")
    save_split(X_test_final,meta_test,y_test,f"data/processed/test(1).csv")
    with open("config/feature_info(1).json","w") as f:
        json.dump({"selected_features":mc_cols,"scaler_path":"models/scaler(1).pkl"},f,indent=4)
    print(f"\nOriginal:{len(feature_cols)}  Final:{len(mc_cols)}")


if __name__== "__main__":
    main()