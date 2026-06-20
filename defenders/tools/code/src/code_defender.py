from defenders.tools.code.src.ast_feature_extractor import extract_security_features_from_ast
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import pandas as pd
import os

def get_embedding(code, tokenizer, model):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embedding = model(**inputs)
    return F.normalize(embedding, p=2, dim=1)


class CodeDefender:
    def __init__(self,deep_check =True,neighbors_number=1,vector_db_path="vector_db.faiss") -> None:
        model_name = "salesforce/codet5p-110m-embedding"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.deep_check = deep_check
        basedir = os.path.dirname(os.path.abspath(__file__))
        self.metadata = pd.read_csv(f"{basedir}/../data/metadata.csv")
        self.vec_db = faiss.read_index(f"{basedir}/../data/{vector_db_path}")
        self.neighbors_number = neighbors_number

    def is_safe(self, code):
        successful, security_issues = extract_security_features_from_ast(code)
        if not successful:
            return False ,"there might be syntax errors in the code"
        if sum(security_issues.values()) == 0:
            return True , "no security issues detected"
        elif not self.deep_check:
            return False , f"security issues detected: {security_issues}"

        vector_features = get_embedding(code, self.tokenizer, self.model).cpu().numpy()
        _, indices = self.vec_db.search(vector_features, k=self.neighbors_number)
        similar_codes = self.metadata.iloc[indices[0]]
        for _, row in similar_codes.iterrows():
            if int(row["label"]) == 1:
                return False , "similar malicious code was found"            
            
        return True,"safe"


# if __name__ == "__main__":
#     defender = CodeDefender(security_level="low", vector_db_path="vector_db.faiss")

#     code = """
# import os
# def sqrt(x):
#     if x < 0:
#         raise ValueError("Cannot compute square root of a negative number.")
#     return x ** 0.5     
# """

#     print("Testing code:")
#     print(code)

#     if defender.is_safe(code):
#         print("\nPrediction: SAFE")
#     else:
#         print("\nPrediction: MALICIOUS")
