import os
import torch
import torch.nn as nn
from torchgen import model
from transformers import AutoTokenizer, DistilBertModel
from torchcrf import CRF
from typing import Counter, List, Tuple, Optional
from defenders.pii_detection.src.pii_crf_predictor import CRFPiiDetector
import fasttext
from defenders.pii_detection.src.utils import tokenise_with_alignment

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


class DistilBERTBiLSTMCRF(nn.Module):

    def __init__(self, num_tags, ignore_index=-100):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(512, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.ignore_index = ignore_index

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask.float()
            )

        lstm_out, _ = self.lstm(bert_out.last_hidden_state)

        emissions = self.classifier(lstm_out)    #four probabilities for each token

        mask = attention_mask.bool()#crf exepect false and true so i use it to convert            

        if labels is not None:#because it use this during train 
            
            safe_labels = labels.clone()
            safe_labels[labels == self.ignore_index] = 0 #crf can not deal with negative label

            loss = -self.crf(emissions, safe_labels, mask=mask, reduction="mean") #return log likelihood so negative so minimize loss = max likelihood
            return loss, emissions

      
        predictions = self.crf.decode(emissions, mask=mask)
        return predictions


class BiLSTMCRF(nn.Module):
    def __init__(self,embedding_dim=300,hidden_size=256,num_classes=2,dropout=0.3):
        super().__init__()

        self.bilstm1 = nn.LSTM(embedding_dim,hidden_size,batch_first=True,bidirectional=True)
        self.bilstm2 = nn.LSTM(hidden_size * 2,hidden_size,batch_first=True,bidirectional=True)
        self.bilstm3 = nn.LSTM(hidden_size * 2,hidden_size,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # map the output of bilstm to number of classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        # to capture the dependencies between output labels
        self.crf = CRF(num_tags=num_classes,batch_first=True)

    def forward(self, x):
        out1, _ = self.bilstm1(x)
        out2, _ = self.bilstm2(out1)
        out3, _ = self.bilstm3(out2)
        # emmision score are logits for each class for each token
        emissions = self.classifier(out3)
        return emissions

    def decode(self, x, mask):
        emissions = self.forward(x)
        prediction = self.crf.decode(emissions,mask=mask)

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
        self.ENWE = fasttext.load_model(os.path.join(_HERE, "..", "models", "cc.en.300.bin"))

        state2 = torch.load(checkpoint_path2, map_location=self.device)
        self.label2id=state2["label2id"]
        self.id2label=state2["id2label"]
        self.model2=BiLSTMCRF(embedding_dim=300,hidden_size=256,num_classes=len(self.label2id))
        self.model2.load_state_dict(state2["model_state_dict"])
        self.model2.to(self.device)
        self.model2.eval()

        self.crf_predictor = CRFPiiDetector()
    
    def trust_strategy(self, predictions1, predictions2,predictions3):
        final_predictions = []
        predictor1_trusted_tags = {"IPV4", "IPV6"}
        predictor3_trusted_tags = {"ACCOUNTNAME", "PASSWORD","EMAIL"}


        for pred1, pred2, pred3 in zip(predictions1, predictions2, predictions3):
            if len(pred1) > 2 and (pred1[2:] in predictor1_trusted_tags):
                final_predictions.append(pred2)
            elif len(pred3) > 2 and (pred3[2:] in predictor3_trusted_tags):
                final_predictions.append(pred3)
            else:
                final_predictions.append(pred1)

        return final_predictions
    
    def voting_strategy(self, predictions1, predictions2, predictions3):
        final_predictions = []

        for pred1, pred2, pred3 in zip(predictions1, predictions2, predictions3):
            votes = [pred1, pred2, pred3]

            final_pred = Counter(votes).most_common(1)[0][0]

            final_predictions.append(final_pred)

        return final_predictions

    def remove_brackets(self,text):
        words = text.split()
        cleaned_words = []
        restricted_chars = ["<",">","<=",">=","{" ,"}","[","]","(",")"]
        for word in words:
            word=word.strip()
            if word in restricted_chars:
                cleaned_words.append(word)
            word = word.lstrip("<").rstrip(">")

            cleaned_words.append(word)
        
        return " ".join(cleaned_words)           

    def predict(self, text) :
        cleaned_text = self.remove_brackets(text)
        words, tokens, subword_ids, word_first_subword = tokenise_with_alignment(self.tokenizer, cleaned_text)
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

        word_tags3 = self.crf_predictor.predict(text)

        final_tags = self.trust_strategy(word_tags,word_tags2,word_tags3)
        return list(zip(words, final_tags))



if __name__ == "__main__":
    checkpoint_path = os.path.join(_HERE, "..", "models", "distilbert_bilstm_crf.pt")
    checkpoint_path2 = os.path.join(_HERE, "..", "models", "pii_ner_model.pth")
    detector = PIIDetector(checkpoint_path,checkpoint_path2)
    text = "My email is johndoe@gmail.com"
    predictions = detector.predict(text)
    print(predictions)