from sentence_transformers import SentenceTransformer

from defenders.multi_turn.TCA.src.feature_extraction import FeatureExtractor
import numpy as np

from defenders.multi_turn.integrated.inference.ssm_feature_extractor import StateFeatureExtractor
from defenders.multi_turn.TCA.inference.toxisty_threat_models import ThreatModel, ToxicityModel

class TestFeatureMath:
    def __init__(self):
        self._embedding_model=SentenceTransformer("all-mpnet-base-v2")
        self._toxicity_model=ToxicityModel()
        self._threat_model=ThreatModel()
        self.extract = FeatureExtractor(toxicity_model=self._toxicity_model,threat_model=self._threat_model,embedding_model=self._embedding_model)
    
        
    def test_zero_drift_for_identical_inputs(self):
        self.extract.reset()
        #see if the values should be zero
        for _ in range(3):
            features = self.extract.extract_features(
                user_msg="Hello",
                prev_assistant_msg="Hi"
            )
        
        assert features["topic_drift_score"] == 0.0
        assert features["drift_acceleration"] == 0.0
        assert features["trajectory_linearity"] == 0.0  

    def test_drift_acceleration_formula(self ):
       
        # Build history
        self.extract.reset()
        self.extract.extract_features("A", "A")
        self.extract.extract_features("B", "B")
        features = self.extract.extract_features("C", "C")

        expected =abs(
            self.extract.drift_history[-1]
            - self.extract.drift_history[-2]
        )
        print (f"Expected drift acceleration: {expected}, Actual: {features['drift_acceleration']}")
        assert abs(features["drift_acceleration"] - expected) < 1e-4  #should be equal

    def test_topic_drift_non_negative(self ):
        self.extract.reset()
        
        features = self.extract.extract_features(
            "Cats are animals",
            "Let's discuss quantum physics"
        )

        assert features["topic_drift_score"] == 0  #firtt message have no drift


    def test_trajectory_linearity_range(self):
        self.extract.reset()
        for text in ["A", "B", "C", "D"]:
            features = self.extract.extract_features(text, text)

        assert 0 <= features["trajectory_linearity"] <= 1 # see values in rage 0 ,1 
        
    def test_trajectory_monotonic_growth(self):
        self.extract.reset()
        last = -1

        for text in ["A", "B", "C", "D"]:
            features = self.extract.extract_features(text, text)
            current = features["trajectory_linearity"]
            print (f"Current: {current}, Last: {last}")
            assert current >= last - 1e-6        #make sure of monotonic growth
            last = current

    
    def test_threat_ordering(self):
        self.extract.reset()
        safe = self.extract.extract_features("Hello friend", "Hi")
        unsafe = self.extract.extract_features("I will kill you", "Hi")

        assert unsafe["threat_score"] >= safe["threat_score"]  # make sure that threat classifier works well
        
    def test_toxicity_ordering(self):
        self.extract.reset()
        safe = self.extract.extract_features("Hello friend", "Hi")
        unsafe = self.extract.extract_features("you are supid","")

        assert unsafe["toxicity_score"] >= safe["toxicity_score"]  # make sure that toxicity classifier works well
        
    def test_repeated_identical_inputs(self):
        self.extract.reset()
        values = []

        for _ in range(5):
            f = self.extract.extract_features("Hello", "Hello")
            values.append(f["topic_drift_score"])

        assert all(v == values[0] for v in values)   #idnetical input then drift should equal 0
        
    

if __name__ == "__main__":
    test =TestFeatureMath()
    test.test_zero_drift_for_identical_inputs()
    test.test_drift_acceleration_formula()
    test.test_topic_drift_non_negative()
    test.test_trajectory_linearity_range()
    test.test_trajectory_monotonic_growth()
    test.test_threat_ordering()
    test.test_toxicity_ordering()