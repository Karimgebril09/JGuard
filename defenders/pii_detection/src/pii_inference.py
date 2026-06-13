import os
import torch
import torch.nn as nn
from torchgen import model
from transformers import AutoTokenizer, DistilBertModel
from torchcrf import CRF
from typing import List, Tuple, Optional
import fasttext

_HERE = os.path.dirname(os.path.abspath(__file__))

NER_TAGS = [
    "B-ACCOUNTNAME",    "B-ACCOUNTNUMBER",   "B-CREDITCARDNUMBER",
    "B-EMAIL",          "B-IPV4",            "B-IPV6",
    "B-MAC",            "B-PASSWORD",        "B-PHONE_NUMBER",
    "B-SSN",            "B-USERNAME",
    "I-ACCOUNTNAME",    "I-ACCOUNTNUMBER",   "I-CREDITCARDNUMBER",
    "I-EMAIL",          "I-IPV4",            "I-IPV6",
    "I-MAC",            "I-PASSWORD",        "I-PHONE_NUMBER",
    "I-SSN",            "I-USERNAME",
    "O",
]
TAG2IDX = {tag: i for i, tag in enumerate(NER_TAGS)}
IDX2TAG = {i: tag for tag, i in TAG2IDX.items()}
NUM_TAGS = len(NER_TAGS)

def mask_pii(text, tagged):

    output_words   = []
    for word, tag in tagged:
        if tag.startswith("B-") or tag.startswith("I-"):
            label= tag[2:]
            output_words.append( f"<{label}>")
        else:
            output_words.append(word)

    return " ".join(output_words)

class DistilBERTBiLSTMCRF(nn.Module):
    def __init__(self, num_tags: int = NUM_TAGS, ignore_index: int = -100):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.lstm       = nn.LSTM(input_size=768, hidden_size=256,
                                  batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, num_tags)
        self.crf        = CRF(num_tags, batch_first=True)
        self.ignore_index = ignore_index

    def forward(self, input_ids, attention_mask, labels=None):
        bert_out        = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask.float())
        lstm_out, _     = self.lstm(bert_out.last_hidden_state)
        emissions       = self.classifier(lstm_out)
        mask            = attention_mask.bool()

        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[labels == self.ignore_index] = 0
            loss = -self.crf(emissions, safe_labels, mask=mask, reduction="mean")
            return loss, emissions

        return self.crf.decode(emissions, mask=mask)


class BiLSTMCRF(nn.Module):
    def __init__(self,embedding_dim=300,hidden_size=256,num_classes=2,dropout=0.3):
        super().__init__()

        self.bilstm1 = nn.LSTM(embedding_dim,hidden_size,batch_first=True,bidirectional=True)
        self.bilstm2 = nn.LSTM(hidden_size * 2,hidden_size,batch_first=True,bidirectional=True)
        self.bilstm3 = nn.LSTM(hidden_size * 2,hidden_size,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.crf = CRF(num_tags=num_classes,batch_first=True)

    def forward(self, x):

        out1, _ = self.bilstm1(x)
        out2, _ = self.bilstm2(out1)
        out3, _ = self.bilstm3(out2)

        emissions = self.classifier(out3)

        return emissions

    def decode(self, x, mask):

        emissions = self.forward(x)

        prediction = self.crf.decode(
            emissions,
            mask=mask
        )

        return prediction


class PIIDetector:
    def __init__(self, checkpoint_path: str,checkpoint_path2: str, device= None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.model = DistilBERTBiLSTMCRF(num_tags=NUM_TAGS)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # self.ENWE = fasttext.load_model("./../models/cc.en.300.bin")
        self.ENWE = fasttext.load_model(os.path.join(_HERE, "..", "models", "cc.en.300.bin"))

        state2 = torch.load(checkpoint_path2, map_location=self.device)
        self.label2id=state2["label2id"]
        self.id2label=state2["id2label"]
        self.model2=BiLSTMCRF(embedding_dim=300,hidden_size=256,num_classes=len(self.label2id))
        self.model2.load_state_dict(state2["model_state_dict"])
        self.model2.to(self.device)
        self.model2.eval()

    
    def trust_strategy(self, predictions1, predictions2):
        # trust model2 in ip addresses
        final_predictions = []

        for pred1, pred2 in zip(predictions1, predictions2):
            if "IPV4" in pred2 or "IPV6" in pred2:
                final_predictions.append(pred2)
            else:
                final_predictions.append(pred1)

        return final_predictions
            

    def predict(self, text) :
        words, tokens, subword_ids, word_first_subword = self.tokenise_with_alignment(text)
        if not subword_ids:
            return []

        input_ids= torch.tensor([subword_ids], dtype=torch.long).to(self.device)
        attn_mask= torch.ones_like(input_ids).to(self.device)
        with torch.no_grad():
            predictions = self.model(input_ids=input_ids, attention_mask=attn_mask)
        subword_tags = predictions[0] 

        with torch.no_grad():
            x = torch.tensor([[self.ENWE.get_word_vector(tok) for tok in tokens]],dtype=torch.float32).to(self.device)
            length = len(tokens)
            mask = torch.ones((1, length), dtype=torch.bool, device=self.device)
            pred = self.model2.decode(x, mask)[0]

        predictions2 = [self.label2id.get(self.id2label[idx], self.label2id["O"]) for idx in pred]
        word_tags = [IDX2TAG[subword_tags[idx]] for idx in word_first_subword]
        word_tags2 = [self.id2label[predictions2[idx]] for idx in word_first_subword]

        final_tags = self.trust_strategy(word_tags,word_tags2)
        return list(zip(words, final_tags))



    def tokenise_with_alignment(self, text):
        words= text.strip().split()
        subword_ids= []
        word_first_subword = []
        tokens = []

        for word in words:
            pieces = self.tokenizer.tokenize(word)
            if not pieces:
                pieces = [self.tokenizer.unk_token]
            ids = self.tokenizer.convert_tokens_to_ids(pieces)
            tokens.extend(pieces)

            word_first_subword.append(len(subword_ids))
            subword_ids.extend(ids)

        return words, tokens, subword_ids, word_first_subword



