import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV,VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier


TRANSFORMS = {
    "toxicity_score":"log1p",
    "threat_score":"log1p",
    "toxicity_score_ema3":"log1p",
    "toxicity_score_rolling3_mean":"log1p",
    "toxicity_score_rolling3_max":"log1p",
    "threat_score_ema3":"log1p",
    "threat_score_rolling3_mean":"log1p",
    "threat_score_rolling3_max":"log1p",
    "toxicity_diff":"square",
    "threat_diff":"square",
    "toxicity_accel":"square",
    "max_toxicity_so_far":"log1p",
    "max_threat_so_far":"log1p",
    "risk_growth_ratio":"log1p",
    "interaction_risk":"log1p",
    "interaction_risk_ema3":"log1p",
    "interaction_risk_rolling3_mean":"log1p",
    "interaction_risk_rolling3_max":"log1p",
    "topic_drift_score":"binarize",
    "cumulative_drift":"binarize",
    "pattern_risk":"binarize",
    "pattern_risk_ema3":"binarize",
    "pattern_risk_rolling3_mean":"binarize",
    "pattern_risk_rolling3_max":"binarize",
    "prev_progressive":"binarize",
    "risk_slope_3":"square",
    "mean_risk_so_far":"log1p",
    "refusal_seen_so_far":"log1p",
    "after_refusal_flag":"log1p",
    "post_refusal":"log1p",
}




os.makedirs("data/processed",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("config",exist_ok=True)


def conversation_split(df):
    all_conv_ids= df["conv_id"].unique()

    train_val_ids,test_ids= train_test_split(all_conv_ids,test_size=0.2,random_state=42,shuffle=True)
    train_ids,val_ids= train_test_split(train_val_ids,test_size=0.25,random_state=42,shuffle=True)

    train_df= df[df["conv_id"].isin(train_ids)].reset_index(drop=True)
    val_df= df[df["conv_id"].isin(val_ids)].reset_index(drop=True)
    test_df= df[df["conv_id"].isin(test_ids)].reset_index(drop=True)

    return {"train":train_df,"val":val_df,"test":test_df}


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
            if corr_col >= corr_other_col:
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
    return series

def apply_transform(df) :
    df = df.copy()
    for feature, transform in TRANSFORMS.items():
        if feature not in df.columns:
            continue
        df[f"{feature}"] = transform_feature(df[feature], transform)
    return df

def load_splits():
    train_df= pd.read_csv("data/processed/train.csv")
    val_df= pd.read_csv("data/processed/validation.csv")
    test_df= pd.read_csv("data/processed/test.csv")

    feature_cols= [c for c in train_df.columns if c not in ["conv_id","turn_id","label"]]

    X_train= train_df[feature_cols]
    X_val= val_df[feature_cols]
    X_test= test_df[feature_cols]
    

    y_train= train_df["label"]
    y_val= val_df["label"]
    y_test= test_df["label"]
    

    meta_train= train_df[["conv_id","turn_id"]]
    meta_val= val_df[["conv_id","turn_id"]]
    meta_test= test_df[["conv_id","turn_id"]]

    # X_train= apply_transform(train_df)
    # X_val= apply_transform(val_df)
    # X_test= apply_transform(test_df)

    print(f"Train:{X_train.shape}  pos={y_train.mean():.3f}")
    print(f"Val:{X_val.shape}  pos={y_val.mean():.3f}")
    print(f"Test:{X_test.shape}  pos={y_test.mean():.3f}")

    return X_train,X_val,X_test,y_train,y_val,y_test,meta_train,meta_val,meta_test,feature_cols


def main():
    df= pd.read_csv("data/total/features_before_selection.csv")
    print("Loaded:",df.shape)

    feature_cols= [c for c in df.columns if c not in ["conv_id","turn_id","label"]]

    splits= conversation_split(df)
    X_train_raw= splits["train"][feature_cols].reset_index(drop=True)
    X_val_raw= splits["val"][feature_cols].reset_index(drop=True)
    X_test_raw= splits["test"][feature_cols].reset_index(drop=True)

    y_train= splits["train"]["label"].reset_index(drop=True)
    y_val= splits["val"]["label"].reset_index(drop=True)
    y_test= splits["test"]["label"].reset_index(drop=True)
    meta_train= splits["train"][["conv_id","turn_id"]].reset_index(drop=True)
    meta_val= splits["val"][["conv_id","turn_id"]].reset_index(drop=True)
    meta_test= splits["test"][["conv_id","turn_id"]].reset_index(drop=True)


    #first scale using robust because of the skewness
    scaler= RobustScaler()
    X_train_scaled= pd.DataFrame(scaler.fit_transform(X_train_raw),columns=feature_cols)
    X_val_scaled= pd.DataFrame(scaler.transform(X_val_raw),columns=feature_cols)
    X_test_scaled= pd.DataFrame(scaler.transform(X_test_raw),columns=feature_cols)
    joblib.dump(scaler,"models/scaler.pkl")


    var_sel= VarianceThreshold(threshold=0.05)
    var_sel.fit(X_train_scaled)
    var_cols= X_train_scaled.columns[var_sel.get_support()].tolist()
    X_train_var= pd.DataFrame(var_sel.transform(X_train_scaled),columns=var_cols)
    X_val_var= pd.DataFrame(var_sel.transform(X_val_scaled),columns=var_cols)
    X_test_var= pd.DataFrame(var_sel.transform(X_test_scaled),columns=var_cols)
    print(f"After variance filter:{len(var_cols)}")


    corr_vals= X_train_var.corrwith(y_train).abs()
    corr_cols= corr_vals[corr_vals > 0.005].index.tolist()
    X_train_corr= X_train_var[corr_cols]
    X_val_corr= X_val_var[corr_cols]
    X_test_corr= X_test_var[corr_cols]
    print(f"After correlation filter:{len(corr_cols)}")

    mc_cols= remove_multicollinear(X_train_corr,y_train,0.9)


    X_train_mc= X_train_corr[mc_cols]
    X_val_mc= X_val_corr[mc_cols]
    X_test_mc= X_test_corr[mc_cols]
    print(f"After multicollinearity:{len(mc_cols)}")


    rfecv= RFECV(
        estimator=DecisionTreeClassifier(max_depth=7,min_samples_leaf=5,
                                         min_samples_split=10,random_state=42),
        step=1,cv=5,scoring="f1_macro",n_jobs=-1,
    )
    rfecv.fit(X_train_mc,y_train)
    final_cols= X_train_mc.columns[rfecv.support_].tolist()
    X_train_final= X_train_mc[final_cols]
    X_val_final= X_val_mc[final_cols]
    X_test_final= X_test_mc[final_cols]
    print(f"After RFECV:{len(final_cols)}")

    save_split(X_train_final,meta_train,y_train,f"{"data/processed"}/train.csv")
    save_split(X_val_final,meta_val,y_val,f"{"data/processed"}/validation.csv")
    save_split(X_test_final,meta_test,y_test,f"{"data/processed"}/test.csv")
    with open("config/feature_info.json","w") as f:
        json.dump({"selected_features":final_cols,"scaler_path":"models/scaler.pkl"},f,indent=4)
    print(f"\nOriginal:{len(feature_cols)}  Final:{len(final_cols)}")


if __name__== "__main__":
    main()