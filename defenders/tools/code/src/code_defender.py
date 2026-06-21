from defenders.tools.code.src.ast_feature_extractor import extract_security_features_from_ast
from defenders.tools.code.src.embedder import Embedder
from defenders.tools.code.src.vector_db import VectorDatabase
from defenders.tools.code.src.chunker import Chunker


class CodeDefender:
    def __init__(self,deep_check =True,neighbors_number=1,vector_db_path="vector_db.faiss") -> None:
        self.deep_check = deep_check
        if not deep_check:
            return
        
        self.embedder = Embedder()
        self.vec_db = VectorDatabase(index_path=vector_db_path, metadata_path="metadata.csv")
        self.vec_db.load_index()
        self.chunker = Chunker()
        self.neighbors_number = neighbors_number

    
    def is_safe(self, code):
        successful, security_issues = extract_security_features_from_ast(code)
        if not successful:
            return False ,"there might be syntax errors in the code"
        if sum(security_issues.values()) == 0:
            return True , "no security issues detected"
        elif not self.deep_check:
            return False , f"security issues detected: {security_issues}"

        chunks=self.chunker.chunk_code(code)
        chunks_embeddings = self.embedder.embedd_chunks(chunks)
        for chunk_emb in chunks_embeddings:
            chunk_results = self.vec_db.search(chunk_emb, k=self.neighbors_number)
            for result in chunk_results:
                if int(result["label"]) == 1:
                    return False , "similar malicious code was found"

        return True, "safe"
      

