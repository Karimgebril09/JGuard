import torch
import pandas as pd
import joblib
from collections import deque
import numpy as np
from defenders.multi_turn.NBF.inference.ssm_model import ConversationDynamicsModel
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "..", "..","integrated", "models")

FEATURE_NAMES = [
            "state_drift", 
            "state_input_distance",
            "long_term_state_drift",
            "state_similarity",
            "state_input_similarity",
            "long_term_state_similarity", 
            "mean_squared_change_short",
            "mean_squared_change_long",
        ]


FEATURES_TRANSFORMATIONS = [
    'long_term_state_drift_sqrt', 
    'mean_squared_change_long_sqrt', 
    'state_input_distance_sqrt', 
    'long_term_state_similarity_sqrt',
    'state_similarity_log',
]

    
class StateFeatureExtractor:
    def __init__(self,embedding_model):
        self.embedding_model = embedding_model
        self.pca_model= joblib.load(os.path.join(_MODELS_DIR, "pca_model_new_labels.pkl"))
        self.ssm_model = ConversationDynamicsModel()
    
        self.x_prev=torch.zeros(1, self.ssm_model.get_state_dim())
        self.x_prev_4step_back = None
        self.q=deque(maxlen=4)

    def embed_input(self, prompt):
        u = self.embedding_model.encode(prompt)
        u = torch.tensor(u, dtype=torch.float32)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        return u
    
    def get_long_term_state(self, x_curr):
        if len(self.q) == 0:
            self.x_prev_4step_back = x_curr
        else:
            self.x_prev_4step_back = self.q[0]
        self.q.append(x_curr)
        return self.x_prev_4step_back
    

    def get_drift(self, x, y):
        if x.dim() == 1:
            x=x.unsqueeze(0)
        if y.dim() == 1:
            y=y.unsqueeze(0)
        return torch.norm(x - y).item()
    
    def get_similarity(self, x, y):
        if x.dim() == 1:
            x=x.unsqueeze(0)
        if y.dim() == 1:
            y=y.unsqueeze(0)
        
        dot = (x * y).sum(dim=1)
        x_norm = torch.sqrt((x * x).sum(dim=1))
        y_norm = torch.sqrt((y * y).sum(dim=1))

        eps = 1e-8  
        cosine = dot / (x_norm * y_norm + eps)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        return cosine.item()
    
    def get_mean_squared_change(self, x, y):
        if x.dim() == 1:
            x=x.unsqueeze(0)
        if y.dim() == 1:
            y=y.unsqueeze(0)

        squared_diff = (x - y) ** 2
        return torch.mean(squared_diff).item()
    

    def reduce_dimensionality(self, vectors):
        vectors = self.pca_model.transform(vectors.numpy())
        vectors_columns=[str(i) for i in range(vectors.shape[1])]   
        vectors = pd.DataFrame(vectors, columns=vectors_columns)
        return vectors


    def apply_selected_transformations(self,df, selected_features):
        transformed = {}

        for feature in selected_features:
            transform = feature.split("_")[-1]
            base_feature = "_".join(feature.split("_")[:-1])
            x = df[base_feature].copy()

            if transform == "orig":
                transformed[base_feature] = x
            else:
                x = x + 1 + 1e-6
                if transform == "log":
                    transformed[feature] = np.log(x)
                elif transform == "sqrt":
                    transformed[feature] = np.sqrt(x)
                else:
                    raise ValueError(f"Unknown transformation: {transform}")

        return pd.DataFrame(transformed)

    def feature_engineering(self, x_t, u_t, x_prev, x_prev_4step_back=None):
        features = []
    
        # drift features
        state_drift = self.get_drift(x_t, x_prev)
        features.append(state_drift)

        state_input_distance = self.get_drift(x_t, u_t)
        features.append(state_input_distance)
            
        long_term_state_drift = self.get_drift(x_t, x_prev_4step_back)
        features.append(long_term_state_drift)

        # similariti features
        state_similarity = self.get_similarity(x_t, x_prev)
        features.append(state_similarity)

        state_input_similarity = self.get_similarity(x_t, u_t)
        features.append(state_input_similarity)


        long_term_state_similarity = self.get_similarity(x_t, x_prev_4step_back)
        features.append(long_term_state_similarity)

        # mean square change feature
        short_term_mse = self.get_mean_squared_change(x_t, x_prev)
        features.append(short_term_mse)

        long_term_mse = self.get_mean_squared_change(x_t, x_prev_4step_back)
        features.append(long_term_mse)

        return torch.tensor(features, dtype=torch.float32)
    
    
    def get_features_transformed(self, features):
        features = pd.DataFrame([features.numpy()], columns=FEATURE_NAMES)
        transformed_selected_features = FEATURES_TRANSFORMATIONS
        features_transformed = self.apply_selected_transformations(features, transformed_selected_features)
        return features_transformed

    
    def extract_features(self, prompt):
        u = self.embed_input(prompt)
        x_curr = self.ssm_model.get_next_state(u)
        self.x_prev_4step_back = self.get_long_term_state(x_curr)
        features = self.feature_engineering(x_curr, u, self.x_prev, self.x_prev_4step_back)
        self.x_prev = x_curr
        vectors=torch.concat([x_curr, u], dim=1)
        vectors= self.reduce_dimensionality(vectors)
        features_transformed = self.get_features_transformed(features)
        return features_transformed, vectors
        