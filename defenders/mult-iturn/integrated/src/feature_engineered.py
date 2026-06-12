import json
import os
import numpy as np
import pandas as pd
from risk_calculator import RiskCalculator


os.makedirs( "data/processed", exist_ok=True)
os.makedirs("reports/figures",   exist_ok=True)
os.makedirs(os.path.dirname("models/scaler.pkl"), exist_ok=True)

def recompute_risks(df, calc):
    df= df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
    interaction_list= []
    pattern_list= []
    progressive_list=[]
    prev_prog_list=[]

    for _cid, group in df.groupby("conv_id", sort=False):
        prev= 0.0
        for _, row in group.iterrows():
            features= row.to_dict()
            interaction= calc.compute_interaction_risk(features)
            pattern= calc.compute_pattern_risk(features)
            prog= calc.calculate_progressive_risk(features, prev)
            interaction_list.append(interaction)
            pattern_list.append(pattern)
            progressive_list.append(prog)
            prev_prog_list.append(prev)
            prev= prog

    df["interaction_risk"]= interaction_list
    df["pattern_risk"]= pattern_list
    df["progressive_risk"]= progressive_list
    df["prev_progressive"]= prev_prog_list
    return df


def add_smoothing_features(df):
    for col in ["toxicity_score", "threat_score", "interaction_risk", "pattern_risk"]:
        df[f"{col}_ema3"]= (
            df.groupby("conv_id")[col]
            .transform(lambda x: x.ewm(span=3, adjust=False).mean())
        )
        df[f"{col}_rolling3_mean"]= (
            df.groupby("conv_id")[col]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        df[f"{col}_rolling3_max"]= (
            df.groupby("conv_id")[col]
            .transform(lambda x: x.rolling(3, min_periods=1).max())
        )
    return df
def add_escalation_features(df: pd.DataFrame) -> pd.DataFrame:
    df["toxicity_diff"]= df.groupby("conv_id")["toxicity_score"].diff().fillna(0)
    df["threat_diff"]= df.groupby("conv_id")["threat_score"].diff().fillna(0)
    df["toxicity_accel"]= df.groupby("conv_id")["toxicity_diff"].diff().fillna(0)
    df["risk_slope_3"]= (
        df.groupby("conv_id")["interaction_risk"]
        .transform(lambda x: x.diff().rolling(3, min_periods=1).mean())
        .fillna(0)
    )
    return df
def add_context_features(df ):
    g= df.groupby("conv_id")
    df["max_toxicity_so_far"]= g["toxicity_score"].cummax()
    df["max_threat_so_far"]= g["threat_score"].cummax()
    df["mean_risk_so_far"]= (
        g["interaction_risk"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_shape_features(df):
    g= df.groupby("conv_id")
    df["early_high_risk"]= g["interaction_risk"].transform(lambda x: x.head(3).max())
    df["late_risk_increase"]= g["interaction_risk"].transform(lambda x: x.tail(3).mean())
    df["risk_growth_ratio"]= df["late_risk_increase"] / (df["early_high_risk"] + 1e-6)
    return df
def build_features(df) :
    df= df.sort_values(["conv_id", "turn_id"])
    df= add_smoothing_features(df)
    df= add_escalation_features(df)
    df= add_context_features(df)
    df= add_shape_features(df)
    return df
def apply_transform(series, transform) :
    if transform== "log1p":
        return np.log1p(np.maximum(series, 0))
    if transform== "square":
        return np.square(series)
    return series


def find_correlated_features(corr_matrix, threshold= 0.9):
    upper= corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    ignore= {"conv_id", "turn_id"}
    pairs= []
    for col in upper.columns:
        if col in ignore:
            continue
        hi= upper.index[abs(upper[col]) > threshold].tolist()
        for fc in hi:
            if fc not in ignore:
                pairs.append({
                    "feature_A":col,
                    "feature_B":fc,
                    "correlation": round(corr_matrix.loc[col, fc], 3),
                })
    return pd.DataFrame(pairs)

def main():
    df= pd.read_csv("data/primitive/multi_turn_data.csv")
    print("Shape:", df.shape)
    params_path= "config/optimized_params_risk.json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            params= json.load(f)
        print("Using optimised risk params")
    else:
        params= {}
        print("Using default risk params")
    calc= RiskCalculator(**params)
    df= recompute_risks(df, calc)
    df= build_features(df.copy())

    #save data 
    df.to_csv(f"data/total/features_before_selection2.csv", index=False)

if __name__== "__main__":
    main()