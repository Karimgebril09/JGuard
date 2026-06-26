import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
import fasttext
import os
import logging

this_file_path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(this_file_path, "../refusal_logging/")):
    os.makedirs(os.path.join(this_file_path, "../refusal_logging/"))
MODEL_LOGGING_PATH = os.path.join(this_file_path, "../refusal_logging/model_fasttext_training.log")
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(MODEL_LOGGING_PATH),  
        logging.StreamHandler(),  
    ],
)
TRAINING_DATA_PATH = os.path.join(this_file_path, "../data/train.csv")
if not os.path.exists(os.path.join(this_file_path, "../models/")):
    os.makedirs(os.path.join(this_file_path, "../models/"))
MODEL_PATH = os.path.join(this_file_path, "../models/best_model_refusal_fasttext.pth")
EMBEDDING_MODEL_PATH = os.path.join(this_file_path, "../models/cc.en.300.bin")



def remove_non_english_char(text):
    # remove ay 7aga mesh english we mesh white space
    text=re.sub(r'[^A-Za-z\s()]', '',text)
    # remove multi space and replace by only one
    text=re.sub(r"\s+"," ",text)
    return text

class RejectionDataset(Dataset):
    def __init__(self, texts, labels, embedding_model, max_len=300, device="cpu"):
        self.texts = texts
        self.labels = labels
        self.embedding_model = embedding_model
        self.max_len=max([len(text.split()) for text in texts])
        self.device = device
        self.embed_dim = embedding_model.get_dimension()
        self.pad_embedding = torch.zeros(self.embed_dim)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()
        length=len(words)
        embeddings = [torch.tensor(self.embedding_model.get_word_vector(word), dtype=torch.float) for word in words]
        
        while len(embeddings) < self.max_len:
            embeddings.append(self.pad_embedding)

        x = torch.stack(embeddings)
        y = torch.tensor(self.labels[idx], dtype=torch.float)

        return x, length, y


class LSTMClassifier(nn.Module):
    def __init__(self, embed_dim=300, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU() 
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, length):
        packed = nn.utils.rnn.pack_padded_sequence(x,length.cpu(),batch_first=True,enforce_sorted=False)
        packed, _ = self.lstm1(packed)
        packed, (h, c) = self.lstm2(packed)
        out = self.fc1(h[-1])
        out = self.relu(out)
        out = self.fc2(out)
        
        return out.squeeze(1)
    
def evaluate(model, val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for x, lengths, y in tqdm(val_dataloader):
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            logits = model(x, lengths)  
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0) 

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    logging.info(f"val Loss: {avg_loss:.4f} ,val Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


def train(model,train_dataset,val_dataset,batch_size=64,epochs=20,learning_rate=1e-3,patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, lengths, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)      
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc = evaluate(model, val_loader)

        logging.info(f"epoch {epoch+1}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f} ,val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered.")
                model.load_state_dict(torch.load(MODEL_PATH))
                break

    return model




if __name__ == "__main__":
    train_df=pd.read_csv(TRAINING_DATA_PATH)
    train_df["response"] = train_df["response"].fillna("")
    train_df["response"] = train_df["response"].apply(remove_non_english_char)
    train_df=train_df[train_df["response"].fillna("").str.len() > 0]
    train_df, val_df = train_test_split(train_df,test_size=0.1,random_state=42,shuffle=True)
    ENWE = fasttext.load_model(EMBEDDING_MODEL_PATH)
    training_data=RejectionDataset(np.array(train_df["response"]),np.array(train_df["label"]),embedding_model=ENWE)
    val_data=RejectionDataset(np.array(val_df["response"]),np.array(val_df["label"]),embedding_model=ENWE)
    model = LSTMClassifier(embed_dim=300,hidden_dim=256)
    train(model,training_data,val_data,batch_size=64,epochs=2,learning_rate=1e-3,patience=3)