import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import fasttext
from torchcrf import CRF
import torch
import torch.nn as nn
from torchcrf import CRF
import os

this_file_path = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(this_file_path, "../data/train.parquet")
VALIDATION_DATA_PATH = os.path.join(this_file_path, "../data/val.parquet")
MODEL_PATH = os.path.join(this_file_path, "../models/pii_ner_model.pth")
EMBEDDING_MODEL_PATH = os.path.join(this_file_path, "../models/cc.en.300.bin")

class PiiDataset(Dataset):

    def __init__(self, texts, labels, embedding_model, label2id):
        self.embedding_model = embedding_model
        self.label2id = label2id
        self.data = [(t, l) for t, l in zip(texts, labels) if len(t) > 0 and len(l) > 0]
        self.max_len = max(len(sent) for sent, _ in self.data)
        self.embed_dim = embedding_model.get_dimension()
        self.pad_embedding = torch.zeros(self.embed_dim)

        self.pad_label = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, labels = self.data[idx]
        embeddings = [torch.tensor(self.embedding_model.get_word_vector(word),dtype=torch.float32) for word in words]
        encoded_labels = [self.label2id[label] for label in labels ]
        length = len(words)
        while len(embeddings) < self.max_len:
            embeddings.append(self.pad_embedding)

        while len(encoded_labels) < self.max_len:
            encoded_labels.append(self.pad_label)

        x = torch.stack(embeddings)
        y = torch.tensor(encoded_labels, dtype=torch.long)
        return x, length, y
    

class BiLSTMCRF(nn.Module):
    def __init__(self,embedding_dim=300,hidden_size=256,num_classes=2,dropout=0.3):
        super().__init__()
        self.bilstm1 = nn.LSTM(embedding_dim,hidden_size,batch_first=True,bidirectional=True)
        self.bilstm2 = nn.LSTM(hidden_size * 2,hidden_size,batch_first=True,bidirectional=True)
        self.bilstm3 = nn.LSTM(hidden_size * 2,hidden_size,batch_first=True,bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 512),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(512, num_classes))

        self.crf = CRF(num_tags=num_classes,batch_first=True)

    def forward(self, x):
        out1, _ = self.bilstm1(x)
        out2, _ = self.bilstm2(out1)
        out3, _ = self.bilstm3(out2)
        emissions = self.classifier(out3)
        return emissions

    def loss(self, x, tags, mask):
        emissions = self.forward(x)
        # compute el negative log-likelihood loss for CRF
        loss = -self.crf(emissions,tags,mask=mask,reduction="mean")
        return loss

    def decode(self, x, mask):
        emissions = self.forward(x)
        prediction = self.crf.decode(emissions,mask=mask)
        return prediction
    

def evaluate(model, val_dataloader, pad_label=-100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    total_correct = 0
    total_tokens = 0
    total_loss = 0

    with torch.no_grad():
        for x, lengths, y in tqdm(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            mask = (y != pad_label)
            safe_labels = y.clone()
            safe_labels[~mask] = 0
            loss = model.loss(x,safe_labels,mask)
            total_loss += loss.item()
            preds = model.decode(x,mask)

            for i in range(len(preds)):
                pred = torch.tensor(preds[i],device=device)
                true = safe_labels[i][:len(pred)]
                total_correct += (pred == true).sum().item()
                total_tokens += len(pred)

    val_loss = total_loss / len(val_dataloader)
    val_acc = total_correct / total_tokens

    print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")
    return val_loss, val_acc


def train(model,train_dataset,val_dataset,batch_size=64,epochs=20,learning_rate=1e-3,pad_label=-100,patience=3):
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for x, lengths, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            mask = (y != pad_label).bool() #create mask
            safe_labels = y.clone()
            safe_labels[~mask] = 0  #change mask from -100 le 0 3ashan crf doesnt accept mask with -100

            optimizer.zero_grad()
            loss = model.loss(x,safe_labels,mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                preds = model.decode(x,mask)
                for i in range(len(preds)):
                    pred = torch.tensor(preds[i],device=device)
                    true = safe_labels[i][mask[i]] # select only valid values 3ashan el crf bey decode el valid values bas

                    total_correct += (pred == true).sum().item()
                    total_tokens += len(true)

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_tokens

        val_loss, val_acc = evaluate(model,val_loader,pad_label)

        print( f"Epoch {epoch+1} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} Val Acc: {val_acc}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                break

    return model



if __name__ == "__main__":
    train_df=pd.read_parquet(TRAINING_DATA_PATH)
    val_df=pd.read_parquet(VALIDATION_DATA_PATH)
    ENWE = fasttext.load_model(EMBEDDING_MODEL_PATH)
    all_labels = set()

    for labels in train_df["token_entity_labels"]:
        all_labels.update(labels)

    label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    id2label = {idx: label for label, idx in label2id.items()}

    training_data = PiiDataset(np.array(train_df["tokenised_unmasked_text"]), np.array(train_df["token_entity_labels"]),embedding_model=ENWE,label2id=label2id)
    val_data = PiiDataset(np.array(val_df["tokenised_unmasked_text"]),np.array(val_df["token_entity_labels"]),embedding_model=ENWE,label2id=label2id)

    model = BiLSTMCRF(embedding_dim=ENWE.get_dimension(),hidden_size=256,num_classes=len(label2id),dropout=0.3)
    trained_model = train(model,training_data,val_data,batch_size=64,epochs=1,learning_rate=1e-3,pad_label=-100,patience=3)

    checkpoint = {"model_state_dict": model.state_dict(),
        "label2id": label2id,
        "id2label": {v: k for k, v in label2id.items()},
    }
    torch.save(checkpoint, MODEL_PATH)

    
