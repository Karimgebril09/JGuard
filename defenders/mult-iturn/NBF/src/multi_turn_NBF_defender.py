import torch
import joblib
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer



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
        ux_prev = torch.cat([u,x_prev], dim=-1)
        x_curr = self.Fxu(ux_prev)
        ux_curr = torch.cat([u,x_curr], dim=-1)
        zt = self.Gxu(ux_curr)
        return x_curr, zt
    

class MultiTurnNBFDefender():
    def __init__(self,):
        self.state_dim = 768    
        self.input_dim = 768    
        self.output_dim = 768   
        self.hidden_dim_ssm1 = 1200  
        self.hidden_dim_ssm2 = 900  
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ssm = stateSpaceModel(state_dim=self.state_dim, input_dim=self.input_dim,\
                                         hidden_dim1=self.hidden_dim_ssm1, hidden_dim2=self.hidden_dim_ssm2,\
                                          output_dim=self.output_dim)
        
        self.ssm.load_state_dict(torch.load("./../ssm_models/ssm_model_exp5/models_best_ssm.pth")['ssm']) 
        self.ssm.eval()
        self.x_prev=torch.zeros(self.state_dim)

        self.pca = joblib.load("./../pca_model_v2_45mapping.pkl") 
        self.classifier = joblib.load("./svm_model_v2_45mapping.pkl")



    def predict(self, prompt):
        # Predict the next state and observation
        # u = pretrained embedding model encoding of the prompt
        u = self.embedding_model.encode(prompt)
        u = torch.tensor(u, dtype=torch.float32)
        x_curr, zt = self.ssm(self.x_prev, u)
        self.x_prev = x_curr
        features=torch.concat([x_curr, u], dim=0)
        features_pca = self.pca.transform(features.cpu().numpy().reshape(1, -1))
        pred=self.classifier.predict(features_pca)
        return pred