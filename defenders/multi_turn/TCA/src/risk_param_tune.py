import json
import os
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from defenders.multi_turn.TCA.src.risk_calculator import RiskCalculator

optuna.logging.set_verbosity(optuna.logging.WARNING)




def recompute_risks(df, calculator):
    df=df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

    interaction_scores=[]
    pattern_scores=[]
    progressive_scores=[]
    previous_scores=[]

    for _, conversation in df.groupby("conv_id", sort=False):

        previous_progressive=0.0

        for _, row in conversation.iterrows():
            interaction=calculator.compute_interaction_risk(row)
            pattern=calculator.compute_pattern_risk(row)
            progressive=calculator.calculate_progressive_risk(
                row,
                previous_progressive,
            )

            interaction_scores.append(interaction)
            pattern_scores.append(pattern)
            progressive_scores.append(progressive)
            previous_scores.append(previous_progressive)

            previous_progressive=progressive

    df["interaction_risk"]=interaction_scores
    df["pattern_risk"]=pattern_scores
    df["progressive_risk"]=progressive_scores
    df["prev_progressive"]=previous_scores

    return df

def evaluate_parameters(params):

    calculator = RiskCalculator(**params)

    #see the k fold cross validate to evaluat the param
    group_kfold = GroupKFold(n_splits=5) 
    auc_scores = []

    X = df
    y = df["label"]
    groups = df["conv_id"]

    for train_idx, valid_idx in group_kfold.split(X,y,groups):
        fold_df = df.iloc[valid_idx].copy()

        fold_df = recompute_risks(fold_df,calculator)
        try:
            #see area under curve
            auc = roc_auc_score(fold_df["label"],fold_df["progressive_risk"])
            auc_scores.append(auc)
        except:
            auc_scores.append(0.5)

    return float(
        np.mean(auc_scores)
    )



def objective(trial):
    """chose parmeter and test it """
    alpha = trial.suggest_float("alpha",0.2,0.8)
    beta = trial.suggest_float("beta",0.05,1.0 - alpha)
    gamma = (1.0- alpha- beta)

    inter_alpha = trial.suggest_float("inter_alpha",0.0,1.0)
    inter_beta = (1.0- inter_alpha)
    pattern_alpha = trial.suggest_float("pattern_alpha", 0.0, 1.0)
    pattern_beta = (1.0- pattern_alpha)

    params = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "inter_alpha": inter_alpha,
        "inter_beta": inter_beta,
        "pattern_alpha": pattern_alpha,
        "pattern_beta": pattern_beta,
    }

    score = evaluate_parameters( params)

    return score
if __name__ == "__main__":
    df=pd.read_csv("defenders/multi_turn/integrated/data/primitive/multi_turn_data(6).csv")
    study = optuna.create_study(direction="maximize")

    study.optimize( objective,  n_trials=1000,show_progress_bar=True)
    best = study.best_params
    final_params = {
        "alpha": best["alpha"],
        "beta": best["beta"],
        "gamma": (1.0 - best["alpha"] - best["beta"] ),
        "inter_alpha": best["inter_alpha"],
        "inter_beta": ( 1.0- best["inter_alpha"]),
        "pattern_alpha": best["pattern_alpha"],
        "pattern_beta": (  1.0- best["pattern_alpha"]),
    }

    os.makedirs("defenders/multi_turn/integrated/config", exist_ok=True)
    with open("defenders/multi_turn/integrated/config/optimized_params_risk(7).json", "w") as f:
        json.dump(final_params, f, indent=4)

