import os
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
import torch.optim as optim
from sentence_transformers import SentenceTransformer
random.seed(42)
import logging
import warnings
warnings.filterwarnings("ignore")

this_file_path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(this_file_path, "../ssm_logging/")):
    os.makedirs(os.path.join(this_file_path, "../ssm_logging/"))
MODEL_LOGGING_PATH = os.path.join(this_file_path, "../ssm_logging/model_training.log")
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(MODEL_LOGGING_PATH),  
        logging.StreamHandler(),  
    ],
)


def convert_text_conversations_to_tensors(conversations, device):
    emb_model = SentenceTransformer('all-mpnet-base-v2') 
    tensor_conversations = []
    for convo in conversations:
        tensor_convo = []
        for turn in convo:
            score = turn['label']
            user_ip_embedding = emb_model.encode(turn["u"])
            a_emb = emb_model.encode(turn["a"])
            tensor_convo.append({'u': user_ip_embedding, 'y': a_emb, 'score': score})
        tensor_conversations.append(tensor_convo)
    return tensor_conversations


def save_tensors(tensor_conversations, save_path,save_name="all_conversations.pt"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(tensor_conversations, os.path.join(save_path, save_name))


class multi_turn_dataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations
        self.longest_convo = max(len(conv) for conv in conversations)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        convo = self.conversations[idx]
        convo_len = len(convo) 
        user_inputs=torch.stack([torch.tensor(turn['u']) for turn in convo])
        ai_responses=torch.stack([torch.tensor(turn['y']) for turn in convo])
        scores=torch.tensor([turn['score'] for turn in convo])
        pad_size = self.longest_convo - convo_len
        mask=torch.ones(convo_len)

        if pad_size > 0:
            u_pad = torch.zeros((pad_size, *user_inputs.shape[1:]))
            y_pad = torch.zeros((pad_size, *ai_responses.shape[1:]))
            s_pad = torch.zeros(pad_size)

            user_inputs = torch.cat([user_inputs, u_pad], dim=0)
            ai_responses = torch.cat([ai_responses, y_pad], dim=0)
            scores = torch.cat([scores, s_pad], dim=0)
            mask=torch.cat([mask, torch.zeros(pad_size)], dim=0)
        
        return user_inputs, ai_responses, scores, mask


def split_data(dataset,batch_size, validation_split=0.05):
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


class stateSpaceModel(nn.Module):
    def __init__(self,state_dim, input_dim, hidden_dim1,hidden_dim2, output_dim):
        super(stateSpaceModel, self).__init__()
        self.Fxu = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, state_dim)
        )
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


def train_ssm(state_model,train_loader,val_loader, num_epochs=200, save_path="./../ssm_models/",model_name="models_best_ssm.pth",
              ssm_learning_rate=1e-4): 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_model.to(device)
    optimizer_ssm = optim.Adam(state_model.parameters(), lr=ssm_learning_rate)
    loss = nn.MSELoss()    
    best_val_loss_ssm = float('inf')  
    for epoch in range(num_epochs):
        state_model.train()
        total_ssm_loss = 0.0

        for u_batch, z_batch, _ , mask in train_loader:
            u_batch = u_batch.to(device)
            z_batch = z_batch.to(device)
            mask = mask.unsqueeze(-1).to(device)
            batch_size, seq_len, state_dim = u_batch.shape
            # state_dim = state_dim
            x_t = torch.zeros(batch_size, state_dim, device=device)
            predicted_z = []
            # loop on convo turns
            for t in range(seq_len):
                u_t = u_batch[:, t, :]
                x_t, z_t = state_model(x_t, u_t)
                predicted_z.append(z_t)
            
            predicted_z = torch.stack(predicted_z, dim=1)

            ssm_loss = loss(predicted_z * mask, z_batch * mask)
            optimizer_ssm.zero_grad()
            ssm_loss.backward() 
            optimizer_ssm.step()

            total_ssm_loss += ssm_loss.item()
        
        # Validation
        state_model.eval()
        val_total_ssm_loss = 0.0
        with torch.no_grad():
            for u_batch, z_batch, _ , mask in val_loader:
                u_batch = u_batch.to(device)
                z_batch = z_batch.to(device)
                mask = mask.unsqueeze(-1).to(device)
                batch_size, seq_len, state_dim = u_batch.shape

                x_t = torch.zeros(batch_size, state_dim, device=device)

                predicted_z = []
                for t in range(seq_len):
                    u_t = u_batch[:, t, :]
                    x_t, z_t = state_model(x_t, u_t)
                    predicted_z.append(z_t)

                predicted_z = torch.stack(predicted_z, dim=1)
                ssm_loss = loss(predicted_z * mask, z_batch * mask)

                val_total_ssm_loss += ssm_loss.item()
                
        # Average losses
        avg_train_ssm_loss = total_ssm_loss / len(train_loader)
        avg_val_total_ssm_loss = val_total_ssm_loss / len(val_loader)


        logging.info(f"Epoch {epoch + 1}/{num_epochs} Train SSM Loss: {avg_train_ssm_loss} Val SSM Loss: {avg_val_total_ssm_loss}")

        if avg_val_total_ssm_loss < best_val_loss_ssm:
            best_val_loss_ssm = avg_val_total_ssm_loss
            torch.save({'ssm': state_model.state_dict(),}, f"{save_path}/{model_name}")
            logging.info(f"New best model saved with Val SSM Loss: {best_val_loss_ssm:.6f}")



if __name__ == "__main__":
    # params
    CONTINUE_TRAINING=False
    GENERATE_CONVERSATION_TENSORS=False
    
    # prepare data
    if GENERATE_CONVERSATION_TENSORS:
        data_path = os.path.join(this_file_path, "../data/")
        with open(os.path.join(data_path, "new_data.json"), "r", encoding="utf-8") as f:
            combined_conversations = json.load(f)
        converted_conversations = convert_text_conversations_to_tensors(combined_conversations, device=torch.device('cpu'))
        save_tensors(converted_conversations, save_path=os.path.join(this_file_path, "../train_val_embedding_data/"), 
                    save_name="all_conversations_new_data.pt")
    
    train1 = torch.load(os.path.join(this_file_path, "../data/circuit_breakers_actorattack.pt"),
    map_location=torch.device('cpu'))

    train2 = torch.load(os.path.join(this_file_path, "../data/circuit_breakers_others.pt"),
        map_location=torch.device('cpu'))

    train3 = torch.load(os.path.join(this_file_path, "../data/all_conversations_new_data.pt"),
        map_location=torch.device('cpu'))

    original_training_data=train1+train2+train3
   
    training_data = random.sample(original_training_data, len(original_training_data))
    dataset=multi_turn_dataset(training_data)
    train_loader, val_loader = split_data(dataset, batch_size=32)
    state_dim = 768    
    input_dim = 768    
    output_dim = 768   
    hidden_dim_ssm1 = 1200  
    hidden_dim_ssm2 = 900  

    state_model=stateSpaceModel(state_dim=state_dim, input_dim=input_dim, hidden_dim1=hidden_dim_ssm1,\
                                hidden_dim2=hidden_dim_ssm2, output_dim=output_dim)
    
    if CONTINUE_TRAINING:
        checkpoint = torch.load(os.path.join(this_file_path, "../models/models_best_ssm.pth"),
                                 map_location=torch.device('cpu'))
        state_model.load_state_dict(checkpoint['ssm'])
    
    save_path=this_file_path + "/../models/"
    train_ssm(state_model, train_loader, val_loader, num_epochs=200, save_path=save_path,model_name="models_best_ssm_new_data.pth",
                    ssm_learning_rate=1e-4)