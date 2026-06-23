import numpy as np
import pandas as pd

from defenders.multi_turn.TCA.src.feature_engineered import add_context_features, add_escalation_features, add_shape_features, add_smoothing_features 



class TestEscalationFeatureMath:

    def test_diff_and_acceleration(self):
        df = pd.DataFrame({
            "conv_id": [1, 1, 1],
            "turn_id": [1, 2, 3],
            "toxicity_score": [0.1, 0.4, 0.2],
            "threat_score": [0.0, 0.2, 0.5],
            "interaction_risk": [1, 2, 3]
        })

        df = add_escalation_features(df)

        # diff
        assert np.isclose(df.loc[1, "toxicity_diff"], 0.3)

        # acceleration
        assert np.isclose(df.loc[2, "toxicity_accel"], -0.5)

    def test_risk_slope_window(self):
        df = pd.DataFrame({
            "conv_id": [1]*4,
            "turn_id": [1,2,3,4],
            "toxicity_score": [0,0,0,0],
            "threat_score": [0,0,0,0],
            "interaction_risk": [1,2,3,4]
        })

        df = add_escalation_features(df)

        # last 3 diffs: [2,2,2]
        
        assert np.isclose(df.loc[3, "risk_slope_3"], 1.0)
        
    def test_ema_exact_formula(self):
        df = pd.DataFrame({
            "conv_id": [1, 1],
            "turn_id": [1, 2],
            "toxicity_score": [0.0, 1.0],
            "threat_score": [0.0, 1.0],
            "interaction_risk": [0.0, 1.0],
            "pattern_risk": [0.0, 1.0]
        })

        df = add_smoothing_features(df)

        alpha = 0.5
        expected = alpha * 1.0 + (1 - alpha) * 0.0

        assert np.isclose(df.loc[1, "toxicity_score_ema3"], expected)
        
    def test_max_and_mean(self):
        df = pd.DataFrame({
            "conv_id": [1,1,1],
            "turn_id": [1,2,3],
            "toxicity_score": [0.2, 0.9, 0.5],
            "threat_score": [0.1, 0.3, 0.4],
            "interaction_risk": [1,2,3]
        })

        df = add_context_features(df)

        assert df["max_toxicity_so_far"].iloc[-1] == 0.9
        assert np.isclose(df["mean_risk_so_far"].iloc[-1], 2.0)
        
    def test_early_late_growth(self):
        df = pd.DataFrame({
            "conv_id": [1]*6,
            "turn_id": list(range(6)),
            "interaction_risk": [1,2,3,4,5,6],
            "toxicity_score": [0]*6,
            "threat_score": [0]*6
        })

        df = add_shape_features(df)

        # early = max([1,2,3]) = 3
        assert df["early_high_risk"].iloc[0] == 3

        # late mean = [4,5,6] = 5
        assert df["late_risk_increase"].iloc[0] == 5.0

    def test_growth_ratio(self):
        df = pd.DataFrame({
            "conv_id": [1]*6,
            "turn_id": list(range(6)),
            "interaction_risk": [1,2,3,4,5,6],
            "toxicity_score": [0]*6,
            "threat_score": [0]*6
        })

        df = add_shape_features(df)

        early = 3
        late = 5
        expected = (late - early) / (early + 1e-6)

        assert np.isclose(df["risk_growth_ratio"].iloc[0], expected)
        

if __name__ == "__main__":
    test = TestEscalationFeatureMath()
    test.test_diff_and_acceleration()
    test.test_risk_slope_window()
    test.test_ema_exact_formula()
    test.test_max_and_mean()
    test.test_early_late_growth()
    test.test_growth_ratio()