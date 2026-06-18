import json
import os
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PowerTransformer
from transformers import pipeline
from defenders.multi_turn.integrated.inference.toxisty_threat_models import ThreatModel, ToxicityModel
from risk_calculator import RiskCalculator
from transforms import TRANSFORMS
from feature_extraction import FeatureExtractor
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)
_BASE_DIR=os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR=os.path.join(_BASE_DIR, "..", "models")
class TCAFeatures:

    def __init__(self, embedding_model,risk_params_path: str="defenders/multi_turn/integrated/config/optimized_params_risk(6).json"):

        risk_params={}
        if os.path.exists(risk_params_path):
            with open(risk_params_path) as f:
                risk_params=json.load(f)
        self._risk_calc=RiskCalculator(**risk_params)

        self._toxicity_model=ToxicityModel()
        self._threat_model=ThreatModel()
        self._embedding_model=embedding_model

        self._raw_history: List[Dict]=[]
        self.reset()

    def reset(self) -> None:
        self._feature_extractor=FeatureExtractor(
            toxicity_model=self._toxicity_model,
            threat_model=self._threat_model,
            embedding_model=self._embedding_model,
        )
        self._prev_prog=0.0
        self.memory=[]

    def feature_extract(self, user_msg, assistant_msg):

        raw=self._feature_extractor.extract_features(user_msg,assistant_msg)

        interaction_risk=self._risk_calc.compute_interaction_risk(raw)
        pattern_risk=self._risk_calc.compute_pattern_risk(raw)
        progressive_risk=self._risk_calc.calculate_progressive_risk(raw, self._prev_prog)

        row={
            **raw,
            "interaction_risk": interaction_risk,
            "pattern_risk": pattern_risk,
            "progressive_risk": progressive_risk,
            "prev_progressive": self._prev_prog,
        }

        row=self.engineer_features(row)

        raw_row_for_memory={
            **raw,
            "interaction_risk": interaction_risk,
            "pattern_risk": pattern_risk,
            "progressive_risk": progressive_risk,
            "prev_progressive": self._prev_prog,
            "toxicity_diff": row["toxicity_diff"],
        }
        self.memory.append(raw_row_for_memory)

        self._prev_prog=progressive_risk

        if len(self.memory) > 10:
            self.memory.pop(0)
            
        feature_info_path=os.path.join(_BASE_DIR, "..", "config", "feature_info(6).json")
        with open(feature_info_path) as f:
            feature_info=json.load(f)
        selected_features=feature_info["selected_features"]
        features=pd.DataFrame([row])
    
        tca_features_transformed=self.apply_transforms(features[selected_features])
    
        scaler=joblib.load(os.path.join(_MODELS_DIR, "scaler(6).pkl"))
        tca_features_transformed[selected_features]=scaler.transform(tca_features_transformed[selected_features])
        
        return tca_features_transformed
    
    def _apply_transform(self, series, transform):
        if transform=="log1p":
            return np.log1p(np.maximum(series, 0))

        if transform=="square":
            return np.square(series)

        if transform=="binarize":
            return (series > 0).astype(float)

        if transform=="yeo-johnson":
            pt=PowerTransformer(method="yeo-johnson", standardize=False)
            return pt.fit_transform(
                np.asarray(series).reshape(-1, 1)
            ).flatten()

        return series

    def apply_transforms(self, df):
        df=df.copy()

        for feature, transform in TRANSFORMS.items():
            if feature not in df.columns:
                continue

            df[feature]=self._apply_transform(df[feature], transform)

        return df
    def engineer_features(self, row):
        history_rows=self.memory

        tox_vals=[h.get("toxicity_score", 0.0) for h in history_rows] + [row["toxicity_score"]]
        thr_vals=[h.get("threat_score", 0.0) for h in history_rows] + [row["threat_score"]]
        ir_vals=[h.get("interaction_risk", 0.0) for h in history_rows] + [row["interaction_risk"]]
        pr_vals=[h.get("pattern_risk", 0.0) for h in history_rows] + [row["pattern_risk"]]

        alpha=2 / (3 + 1)

        def ema3(vals):
            s=vals[0]
            for v in vals[1:]:
                s=alpha * v + (1 - alpha) * s
            return float(s)

        def roll3_mean(vals):
            return float(np.mean(vals[-3:]))

        def roll3_max(vals):
            return float(np.max(vals[-3:]))

        for name, vals in [("toxicity_score", tox_vals),("threat_score", thr_vals), ("interaction_risk", ir_vals), ("pattern_risk", pr_vals), ]:
            row[f"{name}_ema3"]=ema3(vals)
            row[f"{name}_rolling3_mean"]=roll3_mean(vals)
            row[f"{name}_rolling3_max"]=roll3_max(vals)

        prev_tox=history_rows[-1].get("toxicity_score", 0.0) if history_rows else 0.0
        prev_thr=history_rows[-1].get("threat_score", 0.0) if history_rows else 0.0

        row["toxicity_diff"]=row["toxicity_score"] - prev_tox
        row["threat_diff"]=row["threat_score"] - prev_thr

        prev_tox_diff=history_rows[-1].get("toxicity_diff", 0.0) if history_rows else 0.0
        prev_thr_diff=history_rows[-1].get("threat_diff", 0.0) if history_rows else 0.0

        row["toxicity_accel"]=row["toxicity_diff"] - prev_tox_diff
        row["threat_accel"]=row["threat_diff"] - prev_thr_diff

        if len(ir_vals) <=1:
            row["risk_slope_3"]=0.0
        else:
            diffs=np.diff(ir_vals)
            row["risk_slope_3"]=float(np.mean(diffs[-3:]))

  
        row["max_toxicity_so_far"]=float(max(tox_vals))
        row["max_threat_so_far"]=float(max(thr_vals))
        row["mean_risk_so_far"]=float(np.mean(ir_vals))

        row["early_high_risk"]=float(np.max(ir_vals[:3]))
        row["late_risk_increase"]=float(np.mean(ir_vals[-3:]))

        row["risk_growth_ratio"]=( row["late_risk_increase"] - row["early_high_risk"]) / (row["early_high_risk"] + 1e-6)

        return row

if __name__=="__main__":
    embdedding_model=SentenceTransformer( "all-mpnet-base-v2")
    tca_extractor=TCAFeatures(embdedding_model)
    user_msg="Hello, how are you?"
    assistant_msg="I'm good, thank you!"
    features=tca_extractor.feature_extract(user_msg, assistant_msg)
    print(features)