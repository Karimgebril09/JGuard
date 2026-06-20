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
        #load alpha beta gamma parameters
        if os.path.exists(risk_params_path):
            with open(risk_params_path) as f:
                risk_params=json.load(f)
        #risk cacutor and models
        self._risk_calc=RiskCalculator(**risk_params)
        self._toxicity_model=ToxicityModel()
        self._threat_model=ThreatModel()
        self._embedding_model=embedding_model
        self.memory=[]
        self.reset()
    #if there new conversation
    def reset(self) -> None:
        self._feature_extractor=FeatureExtractor(
            toxicity_model=self._toxicity_model,
            threat_model=self._threat_model,
            embedding_model=self._embedding_model,
        )
        self._prev_prog=0.0
        self.memory=[]

    def feature_extract(self, user_msg, assistant_msg):
        #first extract raw features 
        raw=self._feature_extractor.extract_features(user_msg,assistant_msg)
        #extract risk features
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
        #extract engineered feature 
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
        # keep window of memory
        if len(self.memory) > 10:
            self.memory.pop(0)
            
        feature_info_path=os.path.join(_BASE_DIR, "..", "config", "feature_info(6).json")
        with open(feature_info_path) as f:
            feature_info=json.load(f)
        selected_features=feature_info["selected_features"]
        features=pd.DataFrame([row])
        #apply feature transformation
        tca_features_transformed=self.apply_transforms(features[selected_features])
    
        scaler=joblib.load(os.path.join(_MODELS_DIR, "scaler(6).pkl"))
        #apply scaling
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
        """apply feature transformations """
        df=df.copy()
        print("Applying transforms to features...")

        for feature, transform in TRANSFORMS.items():
            if feature not in df.columns:
                continue

            df[feature]=self._apply_transform(df[feature], transform)

        return df
    def engineer_features(self, row):
        """return engineered features"""
        history = self.memory

        tox_vals = [h.get("toxicity_score", 0.0) for h in history]
        tox_vals.append(row["toxicity_score"])

        thr_vals = [h.get("threat_score", 0.0) for h in history]
        thr_vals.append(row["threat_score"])

        ir_vals = [h.get("interaction_risk", 0.0) for h in history]
        ir_vals.append(row["interaction_risk"])

        pr_vals = [h.get("pattern_risk", 0.0) for h in history]
        pr_vals.append(row["pattern_risk"])

        alpha = 0.5 

        #exponential moving average features
        def ema(values):
            s = values[0]
            for v in values[1:]:
                s = alpha * v + (1 - alpha) * s
            return s
        
        #features based on last 3 turns mean 
        def last3_mean(values):
            return sum(values[-3:]) / len(values[-3:])
        
        #feature based on last 3 turns max
        def last3_max(values):
            return max(values[-3:])

        row["toxicity_ema3"] = ema(tox_vals)
        row["threat_ema3"] = ema(thr_vals)
        row["interaction_risk_ema3"] = ema(ir_vals)
        row["pattern_risk_ema3"] = ema(pr_vals)

        row["toxicity_rolling3_mean"] = last3_mean(tox_vals)
        row["threat_rolling3_mean"] = last3_mean(thr_vals)
        row["interaction_rolling3_mean"] = last3_mean(ir_vals)
        row["pattern_rolling3_mean"] = last3_mean(pr_vals)

        row["toxicity_rolling3_max"] = last3_max(tox_vals)
        row["threat_rolling3_max"] = last3_max(thr_vals)
        row["interaction_rolling3_max"] = last3_max(ir_vals)
        row["pattern_rolling3_max"] = last3_max(pr_vals)

        prev_tox = history[-1]["toxicity_score"] if history else 0.0
        prev_thr = history[-1]["threat_score"] if history else 0.0

        row["toxicity_diff"] = row["toxicity_score"] - prev_tox
        row["threat_diff"] = row["threat_score"] - prev_thr

        prev_tox_diff = history[-1].get("toxicity_diff", 0.0) if history else 0.0
        prev_thr_diff = history[-1].get("threat_diff", 0.0) if history else 0.0

        row["toxicity_accel"] = row["toxicity_diff"] - prev_tox_diff
        row["threat_accel"] = row["threat_diff"] - prev_thr_diff

        if len(ir_vals) > 1:
            diffs = []
            for i in range(1, len(ir_vals)):
                diffs.append(ir_vals[i] - ir_vals[i - 1])

            row["risk_slope_3"] = sum(diffs[-3:]) / len(diffs[-3:])
        else:
            row["risk_slope_3"] = 0.0

        row["max_toxicity_so_far"] = max(tox_vals)
        row["max_threat_so_far"] = max(thr_vals)
        row["mean_risk_so_far"] = sum(ir_vals) / len(ir_vals)

        row["early_high_risk"] = max(ir_vals[:3]) if len(ir_vals) >= 3 else ir_vals[-1]
        row["late_risk_increase"] = sum(ir_vals[-3:]) / min(3, len(ir_vals))

        row["risk_growth_ratio"] = (row["late_risk_increase"] - row["early_high_risk"]) / (row["early_high_risk"] + 1e-6)

        return row

# if __name__=="__main__":
#     embdedding_model=SentenceTransformer( "all-mpnet-base-v2")
#     tca_extractor=TCAFeatures(embdedding_model)
#     user_msg="Hello, how are you?"
#     assistant_msg="I'm good, thank you!"
#     features=tca_extractor.feature_extract(user_msg, assistant_msg)
#     print(features)