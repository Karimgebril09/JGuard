import json
import os
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from transformers import pipeline
from risk_calculator import RiskCalculator
from feature_extraction import FeatureExtractor


class TCAFeatures:

    def __init__(self, embedding_model,risk_params_path: str = "../config/optimized_params_risk.json",
                 toxicity_model_name: str = "facebook/roberta-hate-speech-dynabench-r4-target",
                 threat_model_name: str = "tomh/toxigen_roberta",
                 device: str = "cpu"):
        self.threshold = .005

        risk_params = {}
        if os.path.exists(risk_params_path):
            with open(risk_params_path) as f:
                risk_params = json.load(f)
        self._risk_calc = RiskCalculator(**risk_params)

        self._toxicity_model = pipeline("text-classification", model=toxicity_model_name, device=device)
        self._threat_model = pipeline("text-classification", model=threat_model_name, device=device)
        self._embedding_model = embedding_model

        self._raw_history: List[Dict] = []
        self.reset()

    def reset(self) -> None:
        self._feature_extractor = FeatureExtractor(
            toxicity_model=self._toxicity_model,
            threat_model=self._threat_model,
            embedding_model=self._embedding_model,
        )
        self._prev_prog = 0.0
        self.memory = []

    def feature_extract(self, user_msg, assistant_msg):

        raw = self._feature_extractor.extract_features(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
        )

        interaction_risk = self._risk_calc.compute_interaction_risk(raw)
        pattern_risk = self._risk_calc.compute_pattern_risk(raw)
        progressive_risk = self._risk_calc.calculate_progressive_risk(raw, self._prev_prog)

        row = {
            **raw,
            "interaction_risk": interaction_risk,
            "pattern_risk": pattern_risk,
            "progressive_risk": progressive_risk,
            "prev_progressive": self._prev_prog,
        }

        row = self.engineer_features(row)

        raw_row_for_memory = {
            **raw,
            "interaction_risk": interaction_risk,
            "pattern_risk": pattern_risk,
            "progressive_risk": progressive_risk,
            "prev_progressive": self._prev_prog,
            "toxicity_diff": row["toxicity_diff"],
        }
        self.memory.append(raw_row_for_memory)

        self._prev_prog = progressive_risk

        if len(self.memory) > 10:
            self.memory.pop(0)



        return row

    def engineer_features(self, row):
        history_rows = self.memory

        tox_vals = [h.get("toxicity_score", 0.0) for h in history_rows] + [row["toxicity_score"]]
        thr_vals = [h.get("threat_score", 0.0) for h in history_rows] + [row["threat_score"]]
        ir_vals = [h.get("interaction_risk", 0.0) for h in history_rows] + [row["interaction_risk"]]
        pr_vals = [h.get("pattern_risk", 0.0) for h in history_rows] + [row["pattern_risk"]]

        def ema3(vals):
            s = vals[0]
            alpha = 2 / (3 + 1)
            for v in vals[1:]:
                s = alpha * v + (1 - alpha) * s
            return s

        def roll3_mean(vals):
            return float(np.mean(vals[-3:]))

        def roll3_max(vals):
            return float(np.max(vals[-3:]))

        for name, vals in [("toxicity_score", tox_vals), ("threat_score", thr_vals),
                           ("interaction_risk", ir_vals), ("pattern_risk", pr_vals)]:
            row[f"{name}_ema3"] = ema3(vals)
            row[f"{name}_rolling3_mean"] = roll3_mean(vals)
            row[f"{name}_rolling3_max"] = roll3_max(vals)

        prev_tox = history_rows[-1].get("toxicity_score", 0.0) if history_rows else 0.0
        prev_thr = history_rows[-1].get("threat_score", 0.0) if history_rows else 0.0
        row["toxicity_diff"] = row["toxicity_score"] - prev_tox
        row["threat_diff"] = row["threat_score"] - prev_thr

        prev_tox_diff = history_rows[-1].get("toxicity_diff", 0.0) if history_rows else 0.0
        row["toxicity_accel"] = row["toxicity_diff"] - prev_tox_diff

        row["threat_accel"] = row["threat_diff"] - (history_rows[-1].get("threat_diff", 0.0) if history_rows else 0.0)    

        recent_ir = [h.get("interaction_risk", 0.0) for h in history_rows[-2:]] + [row["interaction_risk"]]
        ir_diffs = [b - a for a, b in zip(recent_ir, recent_ir[1:])]
        row["risk_slope_3"] = float(np.mean(ir_diffs)) if ir_diffs else 0.0

        row["max_toxicity_so_far"] = max([h.get("toxicity_score", 0.0) for h in history_rows] + [row["toxicity_score"]])
        row["max_threat_so_far"] = max([h.get("threat_score", 0.0) for h in history_rows] + [row["threat_score"]])
        row["mean_risk_so_far"] = float(np.mean(ir_vals))

        row["early_high_risk"] = float(np.max(ir_vals[:3]))
        row["late_risk_increase"] = float(np.mean(ir_vals[-3:]))
        row["risk_growth_ratio"] = row["late_risk_increase"] / (row["early_high_risk"] + 1e-6)

       

        return row
