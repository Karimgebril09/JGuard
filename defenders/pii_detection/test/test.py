import pandas as pd 
from transformers import AutoTokenizer
from defenders.pii_detection.src.pii_inference import PIIDetector
from sklearn.metrics import classification_report, accuracy_score
import os
_HERE = os.path.dirname(os.path.abspath(__file__))


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



def evaluate(model, df_test):
    all_predictions = []
    true_labels = []

    for i in range(len(df_test)):
        text = " ".join(df_test["words"].iloc[i])

        predictions = model.predict(text)
        predicted_labels = [label for _, label in predictions]

        gold_labels = df_test["labels"].iloc[i]

        # Skip or handle mismatched lengths
        if len(predicted_labels) != len(gold_labels):
            print(
                f"Sentence {i}: prediction length={len(predicted_labels)}, "
                f"true length={len(gold_labels)}"
            )
            continue

        all_predictions.extend(predicted_labels)
        true_labels.extend(gold_labels)

    accuracy = accuracy_score(true_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(true_labels, all_predictions, zero_division=0))
    return accuracy


if __name__ == "__main__":
    # load parquet file
    checkpoint_path = os.path.join(_HERE, "..", "models", "distilbert_bilstm_crf.pth")
    checkpoint_path2 = os.path.join(_HERE, "..", "models", "pii_ner_model.pth")
    df_test= pd.read_parquet('./defenders/pii_detection/data/test.parquet')
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    df_test=prepare_dataset(df_test, tokenizer)
    model=PIIDetector(checkpoint_path,checkpoint_path2)
    evaluation_accuracy = evaluate(model, df_test)
    print(f"Evaluation Accuracy: {evaluation_accuracy:.4f}")
