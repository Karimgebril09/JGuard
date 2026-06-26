from defenders.tools.code.src.embedder import CodeEmbedder
from defenders.tools.code.src.chunker import CodeChunker
from defenders.tools.code.src.vector_db import VectorDatabase
import pandas as pd
import os
import numpy as np

class VectorDatabaseBuilder:
    def __init__(self, srcs_dir):
        self.embedder = CodeEmbedder()
        self.chunker = CodeChunker()
        self.vector_db = VectorDatabase(index_path="vector_db.faiss", metadata_path="metadata.csv")
        self.vector_db.load_index() # if db already exist we start to augment it
        self.srcs_dir = srcs_dir

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
                chunks= self.chunker.chunk_code(code)
                chunks_emb = self.embedder.embedd_chunks(chunks)
                for chunk_emb in chunks_emb: 
                    # append every chunks with the label
                    vectorfeatures.append(chunk_emb.cpu().numpy())
                    metadata.append({"id": idx,"label": label})
                    idx += 1

        vector_np = np.array(vectorfeatures, dtype=np.float32)
        vector_np = vector_np.reshape(vector_np.shape[0], -1)
        metadata_df = pd.DataFrame(metadata)
        self.vector_db.add_vectors(vector_np, metadata_df)
        self.vector_db.save()

        