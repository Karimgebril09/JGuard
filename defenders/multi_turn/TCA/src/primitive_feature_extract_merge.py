import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from defenders.multi_turn.TCA.src.feature_extraction import FeatureExtractor
from defenders.multi_turn.TCA.inference.toxisty_threat_models import ThreatModel,ToxicityModel
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)

toxicity_model=ToxicityModel()
threat_model=ThreatModel()
embedding_model=SentenceTransformer('all-mpnet-base-v2')
feature_extractor=FeatureExtractor(toxicity_model=toxicity_model,threat_model=threat_model,embedding_model=embedding_model,)


def process_unified(path) :
    with open(path,"r",encoding="utf-8") as f:
        dataset=json.load(f)
    all_rows=[]

    for conv_id,turns in enumerate(dataset):
        feature_extractor.reset()
        prev_response=""
        
        for turn_id,turn in enumerate(turns):
            user_msg=turn["u"]
            label=turn["label"]
            
            features=feature_extractor.extract_features(user_msg=user_msg,prev_assistant_msg=prev_response )
            
            all_rows.append({"conv_id": conv_id,"turn_id": turn_id,"label": label,**features})
            prev_response=turn["a"]  
    return pd.DataFrame(all_rows)

if __name__=="__main__":
    df=process_unified("defenders/multi_turn/integrated/data/raw/new_data.json")
    df.to_csv("defenders/multi_turn/integrated/data/primitive/multi_turn_data(6).csv",index=False)
    # df=process_unified("/kaggle/input/datasets/mariamamin30/multiturn-dataset/combined_conversations.json")
    # df.to_csv("/kaggle/working/multi_turn_data.csv",index=False)