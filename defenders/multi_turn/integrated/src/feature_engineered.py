import json
import os
from fastapi import params
from fastapi import params
import numpy as np
import pandas as pd

from defenders.multi_turn.integrated.inference.risk_calculator import RiskCalculator

def recompute_risks(df, calc):
    df=df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
    interaction_list=[]
    pattern_list=[]
    progressive_list=[]
    prev_prog_list=[]

    for _cid, group in df.groupby("conv_id", sort=False):
        prev=0.0
        for _, row in group.iterrows():
            features=row.to_dict()
            interaction=calc.compute_interaction_risk(features)
            pattern=calc.compute_pattern_risk(features)
            prog=calc.calculate_progressive_risk(features, prev)
            interaction_list.append(interaction)
            pattern_list.append(pattern)
            progressive_list.append(prog)
            prev_prog_list.append(prev)
            prev=prog

    df["interaction_risk"]=interaction_list
    df["pattern_risk"]=pattern_list
    df["progressive_risk"]=progressive_list
    df["prev_progressive"]=prev_prog_list
    return df

def add_escalation_features(df):

    df["toxicity_diff"] = 0.0
    df["threat_diff"] = 0.0
    df["toxicity_accel"] = 0.0
    df["threat_accel"] = 0.0
    df["risk_slope_3"] = 0.0

    for conv_id in df["conv_id"].unique():
        conv_idx = df[df["conv_id"] == conv_id].index
        prev_tox = 0
        prev_thr = 0
        prev_risk = 0
        prev_tox_diff = 0
        prev_thr_diff = 0
        risk_window = []
        first = True
        for idx in conv_idx:

            tox = df.loc[idx, "toxicity_score"]
            thr = df.loc[idx, "threat_score"]
            risk = df.loc[idx, "interaction_risk"]

            tox_diff = tox - prev_tox
            thr_diff = thr - prev_thr

            df.loc[idx, "toxicity_diff"] = tox_diff
            df.loc[idx, "threat_diff"] = thr_diff

            tox_acc = tox_diff - prev_tox_diff
            thr_acc = thr_diff - prev_thr_diff
            df.loc[idx, "toxicity_accel"] = tox_acc
            df.loc[idx, "threat_accel"] = thr_acc

            risk_window.append(risk_diff := risk - prev_risk)

            if len(risk_window) > 3:
                risk_window.pop(0)

            risk_slope_3 = sum(risk_window) / len(risk_window)
            df.loc[idx, "risk_slope_3"] = risk_slope_3    # slope of risk over the last 3 turns so that we can see if it increase that mean danger 

            # update previous values
            prev_tox = tox
            prev_thr = thr
            prev_risk = risk

            prev_tox_diff = tox_diff
            prev_thr_diff = thr_diff

    return df

def add_smoothing_features(df):

    alpha = 0.5  # smoothing factor

    # initialize new columns
    df["toxicity_score_ema3"] = 0.0
    df["threat_score_ema3"] = 0.0
    df["interaction_risk_ema3"] = 0.0
    df["pattern_risk_ema3"] = 0.0

    for conv_id in df["conv_id"].unique():
        conv_df = df[df["conv_id"] == conv_id]
        prev_tox = 0
        prev_thr = 0
        prev_int = 0
        prev_pat = 0
        first = True

        for idx in conv_df.index:
            tox = df.loc[idx, "toxicity_score"]
            thr = df.loc[idx, "threat_score"]
            inter = df.loc[idx, "interaction_risk"]
            pat = df.loc[idx, "pattern_risk"]

            if first: #for the bigein
                tox_ema = tox
                thr_ema = thr
                inter_ema = inter
                pat_ema = pat
                first = False
            else:
                tox_ema = alpha * tox + (1 - alpha) * prev_tox
                thr_ema = alpha * thr + (1 - alpha) * prev_thr
                inter_ema = alpha * inter + (1 - alpha) * prev_int
                pat_ema = alpha * pat + (1 - alpha) * prev_pat

            df.loc[idx, "toxicity_score_ema3"] = tox_ema
            df.loc[idx, "threat_score_ema3"] = thr_ema
            df.loc[idx, "interaction_risk_ema3"] = inter_ema
            df.loc[idx, "pattern_risk_ema3"] = pat_ema

            prev_tox = tox_ema
            prev_thr = thr_ema
            prev_int = inter_ema
            prev_pat = pat_ema
    return df

def add_context_features(df):

    max_tox_l = []
    max_thr_l = []
    mean_risk_l = []

    groups = df.groupby("conv_id")

    for conv_id, group in groups:
        max_toxicity = 0
        max_threat = 0
        risk_sum = 0
        count = 0

        for _, row in group.iterrows():
            if row["toxicity_score"] > max_toxicity:
                max_toxicity = row["toxicity_score"]
            max_tox_l.append(max_toxicity)

            if row["threat_score"] > max_threat:
                max_threat = row["threat_score"]
            max_thr_l.append(max_threat)

            risk_sum += row["interaction_risk"]
            count += 1
            mean_risk = risk_sum / count
            mean_risk_l.append(mean_risk)

    df["max_toxicity_so_far"] = max_tox_l
    df["max_threat_so_far"] = max_thr_l
    df["mean_risk_so_far"] = mean_risk_l

    return df
def add_shape_features(df):

    df["early_high_risk"] = 0.0
    df["late_risk_increase"] = 0.0
    df["risk_growth_ratio"] = 0.0

    for conv_id in df["conv_id"].unique():

        conv_idx = list(df[df["conv_id"] == conv_id].index)

        risks = []

        for idx in conv_idx:
            risks.append(df.loc[idx, "interaction_risk"])

        # first 3 turns max
        early_part = risks[:3]
        early_high = max(early_part) if len(early_part) > 0 else 0

        # last 3 turns mean
        late_part = risks[-3:]
        late_mean = sum(late_part) / len(late_part) if len(late_part) > 0 else 0

        # ratio safely
        growth_ratio = (late_mean - early_high) / (early_high + 1e-6)

        for idx in conv_idx:
            df.loc[idx, "early_high_risk"] = early_high
            df.loc[idx, "late_risk_increase"] = late_mean
            df.loc[idx, "risk_growth_ratio"] = growth_ratio

    return df
def build_features(df) :
    df=df.sort_values(["conv_id", "turn_id"])
    
    df=add_smoothing_features(df)
    
    df=add_escalation_features(df)
    
    df=add_context_features(df)
    df=add_shape_features(df)

    return df

def main():
    df=pd.read_csv("defenders/multi_turn/integrated/data/primitive/multi_turn_data(6).csv")
    print("Shape:", df.shape)
    
    params_path="defenders/multi_turn/integrated/config/optimized_params_risk(6).json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            params=json.load(f)
        print("Using optimised risk params")
    else:
        params={}
        print("Using default risk params")
    calc=RiskCalculator(**params)
    df=recompute_risks(df, calc)
    df=build_features(df.copy())

    #save data 
    df.to_csv(f"defenders/multi_turn/integrated/data/total/features_before_selection(6).csv", index=False)

if __name__=="__main__":
    main()