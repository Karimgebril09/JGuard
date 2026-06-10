import json
import os
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
from risk_calculator import RiskCalculator

optuna.logging.set_verbosity(optuna.logging.WARNING)
df= pd.read_csv("data/processed/multi_turn_data.csv")
conv_ids= df["conv_id"].unique()
train_conv_ids, valid_conv_ids= train_test_split(
    conv_ids,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

train_df= df[df["conv_id"].isin(train_conv_ids)].copy()
valid_df= df[df["conv_id"].isin(valid_conv_ids)].copy()


def recompute_risks(df, calc):
    df= df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

    interaction_list= []
    pattern_list= []
    progressive_list= []
    prev_prog_list= []

    for _, group in df.groupby("conv_id", sort=False):
        prev= 0.0
        for _, row in group.iterrows():
            interaction= calc.compute_interaction_risk(row)
            pattern= calc.compute_pattern_risk(row)
            prog= calc.calculate_progressive_risk(row, prev)

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


def suggest_triplet(trial, prefix):
    a= trial.suggest_float(f"{prefix}_a", 0.0, 1.0)
    b= trial.suggest_float(f"{prefix}_b", 0.0, 1.0 - a)
    c= 1.0 - a - b
    return a, b, c
def suggest_pair(trial, prefix):
    a= trial.suggest_float(f"{prefix}_a", 0.0, 1.0)
    b= 1.0 - a
    return a, b
def correlation_score(y_true, scores):
    corr, _= pointbiserialr(y_true, scores)
    return float(corr) if not np.isnan(corr) else 0.0


def objective(trial: optuna.Trial):
    alpha, beta, gamma= suggest_triplet(trial, "main")
    inter_alpha, inter_beta= suggest_pair(trial, "inter")
    pattern_alpha, pattern_beta= suggest_pair(trial, "pattern")
    params= dict(alpha=alpha,beta=beta,gamma=gamma,inter_alpha=inter_alpha, inter_beta=inter_beta,pattern_alpha=pattern_alpha, pattern_beta=pattern_beta)
    risk_calc= RiskCalculator(**params)
    valid_scored= recompute_risks(valid_df.copy(), risk_calc)

    return correlation_score(valid_scored["label"], valid_scored["progressive_risk"])


if __name__== "__main__":
    study= optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2000, show_progress_bar=True)

    print("\nBest correlation:", study.best_value)
    bp= study.best_params
    alpha= bp["main_a"]
    beta= bp["main_b"]
    gamma= 1.0 - alpha - beta
    inter_alpha= bp["inter_a"]
    inter_beta= 1.0 - inter_alpha
    pattern_alpha= bp["pattern_a"]
    pattern_beta= 1.0 - pattern_alpha

    final_params= {
        "alpha":alpha,
        "beta":beta,
        "gamma":gamma,
        "inter_alpha":inter_alpha,
        "inter_beta":inter_beta,
        "pattern_alpha": pattern_alpha,
        "pattern_beta":pattern_beta,
    }
    print("Best params:", final_params)

    final_calc= RiskCalculator(**final_params)
    valid_scored= recompute_risks(valid_df.copy(), final_calc)

    final_corr= correlation_score(valid_scored["label"], valid_scored["progressive_risk"])

    os.makedirs("config", exist_ok=True)
    with open("config/optimized_params_risk.json", "w") as f:
        json.dump(final_params, f, indent=4)

    print("\nFinal validation correlation:", round(final_corr, 4))
    print("Saved to config/optimized_params_risk.json")