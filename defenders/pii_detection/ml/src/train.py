

import argparse
import pickle
from collections import defaultdict
from ast import literal_eval
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from HMM import HMM
from label_maps import NER_TAG_TO_INDEX,INDEX_TO_NER_TAG,NER_TAGS




def load_data(parquet_path):
    df= pd.read_parquet(parquet_path)
    print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")
    return df

def split_data(df,test_size=0.2,random_state=42):
    train_raw,test_raw= train_test_split(df,test_size=test_size,random_state=random_state)

    train_df= pd.DataFrame({
        "tokens":    train_raw["tokenised_unmasked_text"].values,
        "ner_tags":  train_raw["token_entity_labels"].values,
    })
    test_df= pd.DataFrame({
        "tokens":    test_raw["tokenised_unmasked_text"].values,
        "ner_tags":  test_raw["token_entity_labels"].values,
    })

    print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    return train_df,test_df



def build_vocab(x_train):
    vocab_to_index= {}
    for seq in x_train:
        for word in seq:
            if word not in vocab_to_index:
                vocab_to_index[word]= len(vocab_to_index)
    vocab_to_index["UNK"]= len(vocab_to_index)
    print(f"Vocabulary size: {len(vocab_to_index)}")
    return vocab_to_index
def threshold_data(x,threshold,vocab_to_index):
    word_freq= defaultdict(int)
    for seq in x:
        for word in seq:
            word_freq[word] += 1
    return [
        [word if word_freq[word] >= threshold else "UNK" for word in seq]
        for seq in x
    ]

def encode_sequences(sequences,vocab_to_index) :
    unk_idx= vocab_to_index["UNK"]
    return [
        [vocab_to_index.get(word,unk_idx) for word in seq]
        for seq in sequences
    ]


def _to_list(value):
    if isinstance(value,str):
        return list(literal_eval(value))
    return list(value)
def _get_item(data,idx):
    return data.iloc[idx] if hasattr(data,"iloc") else data[idx]

def predict(hmm_model,x_encoded):
    return [hmm_model.viterbi_algorithm(seq) for seq in x_encoded]

def flatten(sequences) :
    return [item for seq in sequences for item in seq]
def evaluate_model(hmm_model,test_ner_tags,test_tokens,
                   ner_tag_strings,index_to_tag,vocab_to_index):
    n_test= len(test_ner_tags) if not hasattr(test_ner_tags,"__len__") else len(test_ner_tags)

    true_tags_all,predicted_tags_all= [],[]
    overall_correct= overall_total= entity_correct= entity_total= 0

    for idx in range(n_test):
        tags= _to_list(_get_item(test_ner_tags,idx))
        observations= _to_list(_get_item(test_tokens,idx))
        if not observations or not tags:
            continue

        unk_idx= vocab_to_index["UNK"]
        obs_indices= [vocab_to_index.get(obs,unk_idx) for obs in observations]
        predicted_ids= hmm_model.viterbi_algorithm(obs_indices)

        aligned_len= min(len(tags),len(predicted_ids))
        for true_id,pred_id in zip(tags[:aligned_len],predicted_ids[:aligned_len]):
            if true_id in index_to_tag and pred_id in index_to_tag:
                true_label= index_to_tag[true_id]
                pred_label= index_to_tag[pred_id]

                true_tags_all.append(true_label)
                predicted_tags_all.append(pred_label)

                if pred_label== true_label:
                    overall_correct += 1
                overall_total += 1

                if true_label != "O":
                    if pred_label== true_label:
                        entity_correct += 1
                    entity_total += 1

    overall_accuracy= overall_correct / overall_total if overall_total > 0 else 0.0
    entity_accuracy= entity_correct  / entity_total  if entity_total  > 0 else 0.0
    print(f"Overall token accuracy (includes O): {overall_accuracy:.4f}")
    print(f"Entity-only accuracy   (excludes O): {entity_accuracy:.4f}")

    compute_metrics(true_tags_all,predicted_tags_all,ner_tag_strings)
    plot_misclassification_rates(ner_tag_strings,true_tags_all,predicted_tags_all)


def compute_metrics(true_tags,predicted_tags,ner_tag_strings):
    states_no_o= [s for s in ner_tag_strings if s not in ("O","UNK")]

    print("\nOverall Metrics (excluding 'O'):")
    print(classification_report(true_tags,predicted_tags,
                                 zero_division=1,labels=states_no_o))

    report= classification_report(true_tags,predicted_tags,
                                    labels=states_no_o,
                                    output_dict=True,zero_division=1)

    precision_values= [report[s]["precision"] for s in states_no_o]
    recall_values= [report[s]["recall"]    for s in states_no_o]
    f1_values= [report[s]["f1-score"]  for s in states_no_o]
    support_values= [report[s]["support"]   for s in states_no_o]
    state_acc_values= [
        sum(1 for t,p in zip(true_tags,predicted_tags) if t== s and p== s)
        / true_tags.count(s) if true_tags.count(s) > 0 else 0
        for s in states_no_o
    ]

    fig,axes= plt.subplots(nrows=3,ncols=2,figsize=(15,15))
    fig.suptitle("Metrics Comparison for Different Entity Types (excluding 'O')",fontsize=16)

    axes[0,0].bar(states_no_o,precision_values,color="skyblue");axes[0,0].set_title("Precision");axes[0,0].tick_params(axis="x",rotation=90)
    axes[0,1].bar(states_no_o,recall_values,color="lightcoral");axes[0,1].set_title("Recall");axes[0,1].tick_params(axis="x",rotation=90)
    axes[1,0].bar(states_no_o,f1_values,color="lightgreen");axes[1,0].set_title("F1-Score");axes[1,0].tick_params(axis="x",rotation=90)
    axes[2,0].bar(states_no_o,state_acc_values,color="orchid");axes[2,0].set_title("State-wise Accuracy");axes[2,0].tick_params(axis="x",rotation=90)

    support_bars= axes[1,1].bar(states_no_o,support_values,color="gold")
    axes[1,1].set_title("Support");axes[1,1].tick_params(axis="x",rotation=90)
    for bar,sv in zip(support_bars,support_values):
        axes[1,1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,str(int(sv)),
                        ha="center",va="bottom")

    fig.delaxes(axes[2,1])
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    print("\nConfusion Matrix (excluding 'O'):")
    conf_matrix= confusion_matrix(true_tags,predicted_tags,labels=states_no_o)
    plt.figure(figsize=(16,10))
    sns.heatmap(conf_matrix,annot=True,fmt="d",
                xticklabels=states_no_o,yticklabels=states_no_o)
    plt.xticks(rotation=90);plt.yticks(rotation=0)
    plt.xlabel("Predicted");plt.ylabel("True")
    plt.title("Confusion Matrix (excluding 'O')")
    plt.show()


def plot_misclassification_rates(ner_tag_strings,true_tags,predicted_tags):
    states= [s for s in ner_tag_strings if s not in ("O","UNK")]
    misc_counts= {s: 0 for s in states}
    for t,p in zip(true_tags,predicted_tags):
        if t in misc_counts and t != p:
            misc_counts[t] += 1

    misc_rates= {
        s: misc_counts[s] / true_tags.count(s) if true_tags.count(s) > 0 else 0
        for s in states
    }
    sorted_states= sorted(states,key=lambda s: misc_rates[s],reverse=True)

    plt.figure(figsize=(10,6))
    plt.bar(sorted_states,[misc_rates[s] for s in sorted_states],color="coral")
    plt.xticks(rotation=90)
    plt.xlabel("Entity Types");plt.ylabel("Misclassification Rate")
    plt.title("Misclassification Rates for Different Entity Types")
    plt.tight_layout();plt.show()

def save_artifacts(hmm_model,vocab_to_index,output_dir):
    import os
    os.makedirs(output_dir,exist_ok=True)
    with open(f"{output_dir}/hmm_model.pkl","wb") as f:
        pickle.dump(hmm_model,f)
    with open(f"{output_dir}/vocab_to_index.pkl","wb") as f:
        pickle.dump(vocab_to_index,f)
    print(f"Artifacts saved to {output_dir}/")

def main():
    df= load_data('/data/data.parquet')
    train_df,test_df= split_data(df)

    x_train= train_df["tokens"].tolist()
    y_train= [[NER_TAG_TO_INDEX[tag] for tag in seq] for seq in train_df["ner_tags"]]

    x_test= test_df["tokens"].tolist()
    y_test= [[NER_TAG_TO_INDEX[tag] for tag in seq] for seq in test_df["ner_tags"]]

    vocab_to_index= build_vocab(x_train)
    x_train_thresh= threshold_data(x_train,1,vocab_to_index)
    x_train_encoded= encode_sequences(x_train_thresh,vocab_to_index)

    x_test_thresh= [[w if w in vocab_to_index else "UNK" for w in seq] for seq in x_test]
    x_test_encoded= encode_sequences(x_test_thresh,vocab_to_index)

    print("\nTraining HMM…")
    hmm= HMM(states=list(range(len(NER_TAGS))),vocab_to_index=vocab_to_index)
    hmm.train_hmm(y_train,x_train_encoded)
    print("Training complete.")

    print("\nEvaluating…")
    evaluate_model(
        hmm_model=hmm,
        test_ner_tags=y_test,
        test_tokens=x_test,
        ner_tag_strings=NER_TAGS,
        index_to_tag=INDEX_TO_NER_TAG,
        vocab_to_index=vocab_to_index,
    )

    save_artifacts(hmm,vocab_to_index,"./artifacts")


if __name__== "__main__":
    main()