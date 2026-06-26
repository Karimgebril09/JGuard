import pandas as pd 
from transformers import AutoTokenizer
from defenders.pii_detection.src.pii_inference import PIIDetector
from sklearn.metrics import classification_report, accuracy_score
from defenders.pii_detection.src.utils import prepare_dataset
import os
_HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, df_test):
    all_predictions = []
    true_labels = []

    for i in range(len(df_test)):
        text = " ".join(df_test["words"].iloc[i])
        predictions = model.predict(text)
        
        predicted_labels = [label[2:] for _, label in predictions]
        gold_labels = [label[2:] for label in df_test["labels"].iloc[i]]
        if len(predicted_labels) != len(gold_labels):
            continue

        all_predictions.extend(predicted_labels)
        true_labels.extend(gold_labels)

    accuracy = accuracy_score(true_labels, all_predictions)
    print(f"accuracy: {accuracy}")
    print(classification_report(true_labels, all_predictions))
    return accuracy


if __name__ == "__main__":
    # load parquet file
    checkpoint_path = os.path.join(_HERE, "..", "models", "distilbert_bilstm_crf.pt")
    checkpoint_path2 = os.path.join(_HERE, "..", "models", "pii_ner_model.pth")
    df_test= pd.read_parquet('./defenders/pii_detection/data_pii/test.parquet')
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    df_test=prepare_dataset(df_test, tokenizer) # fix dataset to be word based
    model=PIIDetector(checkpoint_path,checkpoint_path2)
    evaluation_accuracy = evaluate(model, df_test)
    print(f"Evaluation Accuracy: {evaluation_accuracy}")
