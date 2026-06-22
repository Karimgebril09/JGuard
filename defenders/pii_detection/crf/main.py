from feature_builder import build_features
from model import LinearCRF
import pandas as pd
from transformers import AutoTokenizer
from defenders.pii_detection.src.utils import prepare_dataset


path_to_data = "./defenders/pii_detection/data/test.parquet"
manager = build_features()
df = pd.read_parquet(path_to_data)
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
df=prepare_dataset(df, tokenizer)
df["tokens"] = df["words"]
df["labels"] = df["labels"]
df = df[["tokens", "labels"]]
df=df.sample(n=50, random_state=42).reset_index(drop=True)
dataset = df.values.tolist()
labels = set()

for _, y in dataset:
    labels.update(y)

labels = sorted(labels)

model = LinearCRF(feature_manager=manager,labels=labels,lr=0.05,epochs=5,l2=1e-4)
model.fit(dataset)
model.save_model("crf_weights.json")

prediction = model.predict(
    [
        "my",
        "email",
        "john@gmail.com",
    ]
)

print(prediction)