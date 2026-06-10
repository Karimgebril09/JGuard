import json
from feature_extraction import FeatureExtractor
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

toxicity_model= pipeline(
    "text-classification",
    model="facebook/roberta-hate-speech-dynabench-r4-target",
    device="cpu",
)

threat_model= pipeline(
    "text-classification",
    model="tomh/toxigen_roberta",
    device="cpu",
)

embedding_model= SentenceTransformer('all-mpnet-base-v2')

feature_extractor= FeatureExtractor(
            toxicity_model=toxicity_model,
            threat_model=threat_model,
            embedding_model=embedding_model,
        
        )

def process_unified(path) :
    with open(path, "r", encoding="utf-8") as f:
        dataset= json.load(f)

    print(f"Found {len(dataset)} conversations")

    all_rows= []
    print("Extracting features...")
    num=0
    for conv_id, turns in enumerate(dataset):
        print(f"Processing conversation {conv_id+1}/{len(dataset)}", end="\r")
        feature_extractor.reset()
       
        for turn_id, turn in enumerate(turns):
            user_msg= turn["u"]
            assistant_msg= turn["a"]
            label= turn["label"]

            features= feature_extractor.extract_features(user_msg=user_msg,assistant_msg=assistant_msg)
        
            all_rows.append({
                "conv_id": conv_id,
                "turn_id": turn_id,
                "label": label,
                **features
            })
        # num+=1
        # if num==10:
        #     break

    return pd.DataFrame(all_rows)

if __name__== "__main__":
    print("Processing unified dataset...")
    df= process_unified("../data/raw/combined_conversations.json")
    df.to_csv("../data/processed/multi_turn_data.csv", index=False)
    print(df.shape)
