import math
from typing import List

class FeatureExtractor:
    def __init__(self, toxicity_model, threat_model, embedding_model):
        self.toxicity_model= toxicity_model
        self.threat_model = threat_model
        self.embedding_model = embedding_model
        self.turn_embeddings = []
    
    def reset(self):
        self.turn_embeddings= []

    def extract_features(
        self,
        user_msg = "",
        assistant_msg = "",
    ) :
        combined_text = f"{user_msg} {assistant_msg}".strip()
        current_embed = self.embed(combined_text) if combined_text else []
        topic_drift_score= self.get_topic_drift(current_embed)
        drift_acceleration = self.get_drift_acceleration(current_embed)

        if current_embed:
            self.turn_embeddings.append(current_embed)

        features = {
            "toxicity_score":self.get_toxicity_score(combined_text),
            "threat_score":self.get_threat_score(combined_text),
            "topic_drift_score":topic_drift_score,
            "drift_acceleration": drift_acceleration,
        }
        return features

    def get_topic_drift(self, current_embed):
        if not current_embed or not self.turn_embeddings:
            return 0.0
        return round(self.cosine_distance(current_embed, self.turn_embeddings[-1]), 4)
    
    def get_drift_acceleration(self, current_embed: List[float]) -> float:
       
        if not current_embed or len(self.turn_embeddings) < 2:
            return 0.0

        recent_drift= self.cosine_distance(current_embed, self.turn_embeddings[-1])
        earlier_drift= self.cosine_distance(self.turn_embeddings[-1], self.turn_embeddings[-2])
        acceleration= recent_drift - earlier_drift        

        return abs(round( acceleration, 4)  )

    
    def get_toxicity_score(self, text):
        if not text:
            return 0.0

        words = text.split()
        scores = []

        for i in range(0, len(words), 250):
            chunk = " ".join(words[i:i + 250])

            result = self.toxicity_model(chunk,truncation=True,max_length=512)[0]
            score = (result["score"] if result["label"] == "hate"else 1.0 - result["score"])
            scores.append(score)
        return max(scores, default=0.0)


    def get_threat_score(self, text):
        if not text or not text.strip():
            return 0.0

        words = text.split()
        scores = []
        for i in range(0, len(words), 250):
            chunk = " ".join(words[i:i + 250])

            result = self.threat_model(chunk,truncation=True,max_length=512,)[0]
            score = (result["score"]if result["label"] == "LABEL_1"else 1.0 - result["score"])

            scores.append(score)
        return max(scores, default=0.0)

    def embed(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    def cosine_distance(self, a, b):
        dot= sum(x * y for x, y in zip(a, b))
        norm_a= math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        similarity = dot / (norm_a * norm_b + 1e-9)
        return 1.0 - similarity