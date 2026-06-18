from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib
from collections import deque
import numpy as np

import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "..", "models")

class stateSpaceModel(nn.Module):
    def __init__(self,state_dim, input_dim, hidden_dim1,hidden_dim2, output_dim):
        super(stateSpaceModel, self).__init__()
        # F(x_{t-1},u_t)=>x_t (Transition model)
        self.Fxu = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, state_dim)
        )

        # G(x_t,u_t)=>z_t (Observation model)
        self.Gxu = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
        
    def forward(self, x_prev, u):
        # gets xt
        ux_prev = torch.cat([u,x_prev], dim=-1)
        x_curr = self.Fxu(ux_prev)

        # gets zt
        ux_curr = torch.cat([u,x_curr], dim=-1)
        zt = self.Gxu(ux_curr)
        return x_curr, zt
    

class StateFeatureExtractor:
    def __init__(self,embedding_model):
        self.state_dim = 768    # x_t
        self.input_dim = 768    # u_t
        self.output_dim = 768   # z_t
        self.hidden_dim_ssm1 = 1200  
        self.hidden_dim_ssm2 = 900  

        self.embedding_model = embedding_model
        self.ssm = stateSpaceModel(self.state_dim, self.input_dim, self.hidden_dim_ssm1, self.hidden_dim_ssm2, self.output_dim)
        # checkpoint = torch.load("./../models/models_best_ssm.pth",map_location=torch.device("cpu"))
        checkpoint = torch.load(os.path.join(_MODELS_DIR, "models_best_ssm_new_data.pth"), map_location=torch.device("cpu"))
        self.ssm.load_state_dict(checkpoint["ssm"])
        self.ssm.eval()

        # self.pca_model= joblib.load("./../models/pca_model.pkl")
        self.pca_model= joblib.load(os.path.join(_MODELS_DIR, "pca_model_new_labels.pkl"))
    
        self.x_prev=torch.zeros(self.state_dim)
        self.x_prev_4step_back = None
        self.q=deque(maxlen=4)

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
        state_drift = torch.norm(x_t -x_prev).item()
        features.append(state_drift)

        state_input_distance = torch.norm(x_t -u_t).item()
        features.append(state_input_distance)
            
        long_term_state_drift = torch.norm(x_t - x_prev_4step_back).item()
        features.append(long_term_state_drift)

        # similarity features
        state_similarity = F.cosine_similarity(x_t.unsqueeze(0),x_prev.unsqueeze(0)).item()
        features.append(state_similarity)

        state_input_similarity = F.cosine_similarity(x_t.unsqueeze(0),u_t.unsqueeze(0)).item()
        features.append(state_input_similarity)


        long_term_state_similarity = F.cosine_similarity(x_t.unsqueeze(0),x_prev_4step_back.unsqueeze(0)).item()
        features.append(long_term_state_similarity)

        features.append(torch.mean((x_t - x_prev) ** 2).item())
        features.append(torch.mean((x_t - x_prev_4step_back) ** 2).item())

        return torch.tensor(features, dtype=torch.float32)
    
    def extract_features(self, prompt):
        u = self.embedding_model.encode(prompt)
        u = torch.tensor(u, dtype=torch.float32)
        x_curr, zt = self.ssm(self.x_prev, u)
        
        if self.x_prev_4step_back is None:
            self.x_prev_4step_back = x_curr

        if len(self.q) == 4:
            self.x_prev_4step_back = self.q[0]
        
        self.q.append(x_curr)
        features = self.feature_engineering(x_curr, u, self.x_prev, self.x_prev_4step_back)
        self.x_prev = x_curr
        vectors=torch.concat([x_curr, u], dim=0)

        vectors = self.pca_model.transform(vectors.detach().cpu().unsqueeze(0).numpy())
        vectors=pd.DataFrame(vectors, columns=[str(i) for i in range(vectors.shape[1])])
        features = pd.DataFrame([features.detach().cpu().numpy()], columns=[
            "state_drift", 
            "state_input_distance",
            "long_term_state_drift",
            "state_similarity",
            "state_input_similarity",
            "long_term_state_similarity", 
            "mean_squared_change_short",
            "mean_squared_change_long",
        ])

        transformed_selected_features = [
            'long_term_state_drift_sqrt', 
            'mean_squared_change_long_sqrt', 
            'state_input_distance_sqrt', 
            'long_term_state_similarity_sqrt',
            'state_similarity_log',
        ]

        features_transformed = self.apply_selected_transformations(features, transformed_selected_features)

        return features_transformed, vectors
        



if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-mpnet-base-v2') 
    extractor = StateFeatureExtractor(embedding_model)

    prompt = "What is the capital of France?"
    features, vectors = extractor.extract_features(prompt)
    print("Extracted features:", features.shape)
    print("Extracted vectors:", vectors.shape)
    print("Features:", features)
    print("Vectors:", vectors)