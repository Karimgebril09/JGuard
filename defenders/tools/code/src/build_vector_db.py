import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import os
import faiss
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embedding(code, tokenizer, model):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embedding = model(**inputs)
    return F.normalize(embedding, p=2, dim=1)


class VectorDatabaseBuilder:
    def __init__(self, srcs_dir):
        model_name = "salesforce/codet5p-110m-embedding"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.srcs_dir = srcs_dir
        self.tokenizer = tokenizer
        self.model = model

    def build_vector_db(self):
        vectorfeatures = []
        metadata = []
        idx = 0

        for src in self.srcs_dir:
            if src.endswith(".parquet"):
                df = pd.read_parquet(src)
            else:
                df = pd.read_csv(src)
            for code, label in zip(df["code"], df["label"]):
                emb = get_embedding(code, self.tokenizer, self.model)
                vectorfeatures.append(emb)
                metadata.append({"id": idx,"label": label})
                idx += 1
            print("finished processing file:", src)
        vector_np = np.array(vectorfeatures, dtype=np.float32)
        vector_np = vector_np.reshape(vector_np.shape[0], -1)
        metadata_df = pd.DataFrame(metadata)

        metadata_df.to_csv("metadata.csv", index=False)
        dim = vector_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vector_np)
        faiss.write_index(self.index, "vector_db.faiss")

        
if __name__ == "__main__":
    srcs_dir = ["./data/train_data_malicious.csv",
                "./data/gemini_extended_python_security_dataset.csv", 
                "./data/function_level_security_dataset.csv",
                "./data/claude_python_security_dataset_extended.csv",
                "./data/benign_dataset_sampled.csv",
                ]
    for src in srcs_dir:
        if not os.path.exists(src):
            print(f"File {src} does not exist. Please check the path.")
            
    print("all files exist, building vector database...")
    builder = VectorDatabaseBuilder(srcs_dir)
    builder.build_vector_db()