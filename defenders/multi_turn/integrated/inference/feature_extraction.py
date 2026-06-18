import math
import numpy as np


class FeatureExtractor:
    MAX_TURNS=10

    def __init__(self, toxicity_model, threat_model, embedding_model):
        self.toxicity_model=toxicity_model
        self.threat_model=threat_model
        self.embedding_model=embedding_model
        self.turn_embeddings=[]
        self.drift_history=[]
        self.origin_embedding=None

    def reset(self):
        self.turn_embeddings=[]
        self.drift_history=[]
        self.origin_embedding=None

    def extract_features(self, user_msg="", prev_assistant_msg=""):
        combined_text=f"{user_msg} {prev_assistant_msg}".strip()
        current_embed=self.embed(combined_text) if combined_text else []

        if current_embed:
            if self.origin_embedding is None:
                self.origin_embedding=current_embed
            self.turn_embeddings.append(current_embed)
            if len(self.turn_embeddings) > self.MAX_TURNS:
                self.turn_embeddings.pop(0)

        topic_drift_score=self.topic_drift(current_embed)
        drift_acceleration=self.drift_acceleration(current_embed)

        if topic_drift_score > 0:
            self.drift_history.append(topic_drift_score)
            if len(self.drift_history) > self.MAX_TURNS:
                self.drift_history.pop(0)

        emb_array=np.array(current_embed) if current_embed else None

        features={
            "toxicity_score": float(self.toxicity_model.score(emb=emb_array)) if emb_array is not None else 0.0,
            "threat_score":float(self.threat_model.score(emb=emb_array))   if emb_array is not None else 0.0,
            "topic_drift_score":topic_drift_score,
            "drift_acceleration": drift_acceleration,
            "origin_drift":self.get_origin_drift(current_embed),
            "drift_momentum": self.drift_momentum(),
            "persistent_drift_count": self.get_persistent_drift_count(),
            "angular_coverage":self.angular_coverage(),
            "distance_ratio": self.distance_ratio(),
            "trajectory_linearity":self.trajectory_linearity(),
            "mean_similarity":self.mean_similarity(),
            "std_similarity": self.std_similarity(),
            "min_similarity": self.min_similarity(),
            "max_similarity": self.max_similarity(),
        }

        return features

    def get_origin_drift(self, current_embed):
        if not current_embed or self.origin_embedding is None:
            return 0.0
        return round(self.cosine_distance(current_embed, self.origin_embedding), 4)
    def get_pairwise_similarities(self):
        if len(self.turn_embeddings) < 2:
            return []
        sims=[]
        for i in range(len(self.turn_embeddings)):
            for j in range(i + 1, len(self.turn_embeddings)):
                dist=self.cosine_distance(self.turn_embeddings[i], self.turn_embeddings[j])
                sims.append(1 - dist)
        return sims
    def mean_similarity(self):
        sims=self.get_pairwise_similarities()
        return round(float(np.mean(sims)), 4) if sims else 0.0

    def std_similarity(self):
        sims=self.get_pairwise_similarities()
        return round(float(np.std(sims)), 4) if sims else 0.0

    def min_similarity(self):
        sims=self.get_pairwise_similarities()
        return round(float(np.min(sims)), 4) if sims else 0.0

    def max_similarity(self):
        sims=self.get_pairwise_similarities()
        return round(float(np.max(sims)), 4) if sims else 0.0

    def angular_coverage(self):
        if len(self.turn_embeddings) < 3:
            return 0.0
        centroid=self.centroid()
        vectors=[]
        for e in self.turn_embeddings:
            vec=np.array(e) - centroid
            vectors.append(vec)
        max_cos=-1

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):

                a=vectors[i]
                b=vectors[j]

                norm_a=np.linalg.norm(a)
                norm_b=np.linalg.norm(b)

                if norm_a==0 or norm_b==0:
                    continue

                dot_product=np.dot(a, b)
                cos_similarity=dot_product / (norm_a * norm_b)

                if cos_similarity > max_cos:
                    max_cos=cos_similarity
            return round(1 - max_cos, 4)

    def distance_ratio(self):
        if len(self.turn_embeddings) < 2:
            return 0.0
        centroid=self.centroid()
        distances=[]
        for e in self.turn_embeddings:
            vec=np.array(e) - centroid
            dist=np.linalg.norm(vec)
            distances.append(dist)
        max_d=max(distances)
        if max_d==0:
            return 0.0
        min_d=min(distances)
        ratio=min_d / max_d
        return round(ratio, 4)

    def trajectory_linearity(self):
        if len(self.turn_embeddings) < 3:
            return 0.0
        X=np.array(self.turn_embeddings)
        X=X - np.mean(X, axis=0)
        cov=np.cov(X.T)
        eigvals=np.sort(np.linalg.eigvalsh(cov))[::-1]
        if len(eigvals) < 2 or eigvals[0] <=0:
            return 0.0
        return round((eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-8), 4)

    def drift_momentum(self):
        if len(self.drift_history) < 2:
            return 0.0
        recent=self.drift_history[-3:]
        return round(sum(recent) / len(recent), 4)
    
    def get_persistent_drift_count(self, threshold=0.15):
        count=0
        for drift in reversed(self.drift_history):
            if drift >=threshold:
                count +=1
            else:
                break
        return count

    def centroid(self):
        if not self.turn_embeddings:
            return None
        return np.mean(np.array(self.turn_embeddings), axis=0)
    
    def topic_drift(self, current_embed):
        
        if not current_embed or len(self.turn_embeddings) < 2:
            return 0.0
        return round(self.cosine_distance(current_embed, self.turn_embeddings[-2]), 4)

    def drift_acceleration(self, current_embed):
        
        if not current_embed or len(self.turn_embeddings) < 3:
            return 0.0
        recent_drift=self.cosine_distance(current_embed, self.turn_embeddings[-2])
        earlier_drift=self.cosine_distance(self.turn_embeddings[-2], self.turn_embeddings[-3])
        return round(abs(recent_drift - earlier_drift), 4)

    def embed(self, text):
        return self.embedding_model.encode(text).tolist()


    def cosine_distance(self, a, b):
        if a is None or b is None:
            return 0.0
        
        if len(a)==0 or len(b)==0:
            return 0.0

        a=np.asarray(a, dtype=np.float64)
        b=np.asarray(b, dtype=np.float64)

        norm_a=np.linalg.norm(a)
        norm_b=np.linalg.norm(b)

        if norm_a==0 or norm_b==0:
            return 0.0

        dot_product=np.dot(a, b)
        cosine_similarity=dot_product / (norm_a * norm_b)
        cosine_similarity=np.clip(cosine_similarity, -1.0, 1.0)
        cosine_distance=1.0 - cosine_similarity

        return float(cosine_distance)