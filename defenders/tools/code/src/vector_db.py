import faiss
import pandas as pd
import numpy as np
import os


class VectorDatabase:
    def __init__(self, index_path: str, metadata_path: str):
        basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        self.index_path = os.path.join(basedir, index_path)
        self.metadata_path = os.path.join(basedir, metadata_path)
        self.index = None
        self.metadata = pd.DataFrame()


    def create_index(self, vectors: np.ndarray):
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        faiss.write_index(self.index, self.index_path)

    
    def add_vectors(self, vectors: np.ndarray, metadata: pd.DataFrame):
        if self.index is None:
            self.create_index(vectors)
        self.metadata = pd.concat([self.metadata, metadata], ignore_index=True)
        faiss.write_index(self.index, self.index_path)

    
    def load_index(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print("index or meta data not found")
            return

        self.index = faiss.read_index(self.index_path)
        self.metadata = pd.read_csv(self.metadata_path)

    
    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            self.metadata.to_csv(self.metadata_path, index=False)

    
    def search(self, query_embedding, k):
        _, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata.iloc[idx])

        return results