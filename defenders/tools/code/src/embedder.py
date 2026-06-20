from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class Embedder:
    def __init__(self,model_name="salesforce/codet5p-110m-embedding"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    
    def embedd(self, code):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            embedding = self.model(**inputs)
        return F.normalize(embedding, p=2, dim=1)
    

    def embedd_chunks(self, chunks):
        chunks_embeddings = []
        for chunk in chunks:
            emb = self.embedd(chunk)
            chunks_embeddings.append(emb)
        return chunks_embeddings