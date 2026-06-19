from feature_builder import build_features
from crf_model import LinearCRF
import pandas as pd
from transformers import AutoTokenizer

def tokenise_with_alignment(text, tokenizer):
    words = text.strip().split()

    subword_ids = []
    tokens = []
    word_first_subword = []

    current_index = 0

    for word in words:
        pieces = tokenizer.tokenize(word)

        if not pieces:
            pieces = [tokenizer.unk_token]

        word_first_subword.append(current_index)

        tokens.extend(pieces)
        subword_ids.extend(tokenizer.convert_tokens_to_ids(pieces))

        current_index += len(pieces)

    return words, tokens, subword_ids, word_first_subword


def prepare_dataset(df, tokenizer):
    all_sentences = []
    
    for i in range(len(df)):
        words, tokens, _, word_first_subword = tokenise_with_alignment(
            df["unmasked_text"].iloc[i],
            tokenizer
        )
    
        subword_labels = df["token_entity_labels"].iloc[i]
    
        sentence_words = []
        sentence_labels = []
    
        for w, idx in zip(words, word_first_subword):
            if idx < len(subword_labels):
                label = subword_labels[idx]
            else:
                label = "O"
    
            sentence_words.append(w)
            sentence_labels.append(label)
    
        # keep only valid sentences
        if len(sentence_words) == len(sentence_labels):
            all_sentences.append([sentence_words, sentence_labels])

    return pd.DataFrame(all_sentences, columns=["words", "labels"])




path_to_data = "./defenders/pii_detection/data/test.parquet"
manager = build_features()
df = pd.read_parquet(path_to_data)
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
df=prepare_dataset(df, tokenizer)
df["tokens"] = df["words"]
df["labels"] = df["labels"]
df = df[["tokens", "labels"]]
df=df.sample(n=200, random_state=42).reset_index(drop=True)
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