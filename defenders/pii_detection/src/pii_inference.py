import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel
from TorchCRF import CRF
from typing import List, Tuple, Optional


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

    def decode(self, x, lengths, mask):

        emissions = self.forward(x, lengths)

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

        self.model2 = DistilBERTBiLSTMCRF()
        state2 = torch.load(checkpoint_path2, map_location=self.device)
        self.label2id=state2["label2id"]
        self.id2label=state2["id2label"]
        self.model2.load_state_dict(state2,map_location=self.device)
        self.model2.to(self.device)
        self.model2.eval()

    def predict(self, text) :
        words, subword_ids, word_first_subword = self.tokenise_with_alignment(text)
        if not subword_ids:
            return []

        input_ids= torch.tensor([subword_ids], dtype=torch.long).to(self.device)
        attn_mask= torch.ones_like(input_ids).to(self.device)
        with torch.no_grad():
            predictions = self.model(input_ids=input_ids, attention_mask=attn_mask)
        subword_tags = predictions[0] 


        # # embeddings to be added to fasttext
        # with torch.no_grad():
        
        #     mask = torch.ones_like(input_ids).to(self.device)
        #     # make it ones mask
        #     # predictions2 = self.model2.crf.decode(x, mask=mask)
        predictions2 = []

        word_tags = [IDX2TAG[subword_tags[idx]] for idx in word_first_subword]
        word_tags2 = [self.id2label[predictions2[0][idx]] for idx in word_first_subword]

        return list(zip(words, word_tags)) , list(zip(words, word_tags2))



    def tokenise_with_alignment(self, text):

        words= text.strip().split()
        subword_ids= []
        word_first_subword = []

        for word in words:
            pieces = self.tokenizer.tokenize(word)
            if not pieces:
                pieces = [self.tokenizer.unk_token]
            ids = self.tokenizer.convert_tokens_to_ids(pieces)

            word_first_subword.append(len(subword_ids))
            subword_ids.extend(ids)

        return words, subword_ids, word_first_subword


    