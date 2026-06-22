import json
import os
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
from defenders.multi_turn.integrated.inference.risk_calculator import RiskCalculator



optuna.logging.set_verbosity(optuna.logging.WARNING)
df=pd.read_csv("defenders/multi_turn/integrated/data/primitive/multi_turn_data(6).csv")
conversation_ids=df["conv_id"].unique()
train_ids, valid_ids=train_test_split(conversation_ids,test_size=0.2,random_state=42,shuffle=True,)

train_df=df[df["conv_id"].isin(train_ids)].copy()
valid_df=df[df["conv_id"].isin(valid_ids)].copy()

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

def correlation_score(labels, scores):
    corr, _=pointbiserialr(labels, scores)

    if np.isnan(corr):
        return 0.0
    return float(corr)


def objective(trial):

    alpha=trial.suggest_float("alpha", 0.0, 1.0)
    beta=trial.suggest_float("beta", 0.0, 1.0 - alpha)
    gamma= 1.0 -alpha - beta
    inter_alpha =trial.suggest_float("inter_alpha", 0.0, 1.0)
    inter_beta =1.0 - inter_alpha
    pattern_alpha= trial.suggest_float("pattern_alpha", 0.0, 1.0)
    pattern_beta=1.0 -  pattern_alpha

    params={
        "alpha":alpha,
        "beta": beta,
        "gamma":gamma,
        "inter_alpha" :inter_alpha,
        "inter_beta":inter_beta,
        "pattern_alpha" :pattern_alpha,
        "pattern_beta":pattern_beta,
    }

    calculator=RiskCalculator(**params)
    scored_df=recompute_risks(valid_df.copy() ,calculator,)
    return correlation_score(scored_df["label"],scored_df["progressive_risk"],)



if __name__=="__main__":

    study=optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=2000 ,show_progress_bar=True,)
    best=study.best_params
    final_params= {
        "alpha":best["alpha"],
        "beta": best["beta"],
        "gamma": 1.0 - best["alpha"] - best["beta"],
        "inter_alpha":best["inter_alpha"],
        "inter_beta":1.0 - best["inter_alpha"],
        "pattern_alpha" :best["pattern_alpha"],
        "pattern_beta" : 1.0 - best["pattern_alpha"],
    }

    print(json.dumps(final_params, indent=4))
    os.makedirs("defenders/multi_turn/integrated/config", exist_ok=True)
    with open("defenders/multi_turn/integrated/config/optimized_params_risk(6).json", "w") as f:
        json.dump(final_params, f, indent=4)
