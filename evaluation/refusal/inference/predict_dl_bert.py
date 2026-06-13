


from matplotlib.pyplot import clf
import re
import torch

from torch import nn
from transformers import DistilBertTokenizerFast, DistilBertModel

def remove_non_english_char(text):
    # remove ay 7aga mesh english we mesh white space
    text=re.sub(r'[^A-Za-z\s()]', '',text)
    #remove multi space and replace by only one
    text=re.sub(r"\s+"," ",text)
    return text


def get_contextualized_embeddings(sentence, model, tokenizer, max_len, device='cuda'):
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_sent_word_embeddings = []
    words = sentence.split()

    encoded = tokenizer(words, return_tensors="pt",is_split_into_words=True,truncation=True,padding="longest")
    for k in encoded:
        encoded[k] = encoded[k].to(device)

    with torch.no_grad():
        outputs = model(**encoded)
    hidden = outputs.last_hidden_state[0]
    word_ids = encoded.word_ids(batch_index=0)

    word_embs = []
    for w_idx in range(len(words)):
        indices = [i for i, wid in enumerate(word_ids) if wid == w_idx]
        if indices:
            emb = hidden[indices].mean(dim=0)
        else:
            emb = torch.zeros(model.config.hidden_size, device=device)
        word_embs.append(emb)

    if len(word_embs) < max_len:
        pad_len = max_len - len(word_embs)
        pad = torch.zeros(model.config.hidden_size, device=device)
        word_embs += [pad] * pad_len
    all_sent_word_embeddings.append(torch.stack(word_embs))
    final_tensor = torch.stack(all_sent_word_embeddings)
    return final_tensor


class LSTMClassifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, dropout=0.3): 
        super().__init__()
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True 
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU() 

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed_input)
        out = h_n[-1] 
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out.squeeze(1)
    

class RefusalInference:
    def __init__(self, embed_dim=768, hidden_dim=256, dropout=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer=DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        self.bert=DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.bert.eval()
        self.model = LSTMClassifier(embed_dim, hidden_dim, dropout)
        self.model.load_state_dict(torch.load("models/lstm_refusal_model_bert.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        cleaned_text=remove_non_english_char(text)
        text = " ".join(cleaned_text.split()[:100])
        emb = get_contextualized_embeddings(text, self.bert, self.tokenizer, max_len=len(text.split()), device=self.device)
        length = torch.tensor([emb.shape[1]])

        with torch.no_grad():
            logit = self.model(emb.to(self.device), length.to(self.device))
            prob = torch.sigmoid(logit).item()

        return {"label": int(prob >= 0.5), "score": prob}
    

if __name__ == "__main__":

    refusal = RefusalInference()
    result = refusal.predict("I can't help with that request.")
    print(result['score'])