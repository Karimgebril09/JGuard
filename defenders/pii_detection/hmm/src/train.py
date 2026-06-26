import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score
from defenders.pii_detection.hmm.src.HMM import HMM
from defenders.pii_detection.hmm.src.word_features import extract_word_features , NER_TAG_TO_INDEX, INDEX_TO_NER_TAG ,NER_TAGS



def encode_tags(tag_seqs):
    return [[NER_TAG_TO_INDEX.get(t, NER_TAG_TO_INDEX['O']) for t in seq] for seq in tag_seqs]

def word_to_feature_indices(word,vocab):
    tokens= [word] + extract_word_features(word)
    return [vocab.get(t, vocab['UNK']) for t in tokens]


def encode_words_enhanced(word_seqs, vocab):
    return [[word_to_feature_indices(w, vocab) for w in seq] for seq in word_seqs]


def main():
    df= pd.read_parquet('../data/word_based_data.parquet')

    train_df, test_df= train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df= train_test_split(train_df, test_size=0.1, random_state=42)
    print(len(train_df), len(val_df), len(test_df))

    train_words_raw=[list(x) for x in train_df['words'].tolist()]
    train_tags_raw= [list(x) for x in train_df['labels'].tolist()]

    test_words_raw= [list(x) for x in test_df['words'].tolist()]
    test_tags_raw=[list(x) for x in test_df['labels'].tolist()]

    train_tags=encode_tags(train_tags_raw)
    test_tags= encode_tags(test_tags_raw)

    # build enhanced vocabulary 
    enhanced_vocab= {}


    for seq in train_words_raw:
        for w in seq:
            if w not in enhanced_vocab:
                enhanced_vocab[w]=len(enhanced_vocab)
            for f in extract_word_features(w):
                if f not in enhanced_vocab:
                    enhanced_vocab[f]= len(enhanced_vocab)

    enhanced_vocab['UNK']= len(enhanced_vocab)

    train_obs_enhanced= encode_words_enhanced(train_words_raw, enhanced_vocab)
    test_obs_enhanced=encode_words_enhanced(test_words_raw,enhanced_vocab)


    print(f"Enhanced vocab size: {len(enhanced_vocab)}")

    enhanced_model= HMM(num_states=len(NER_TAGS), vocab_size=len(enhanced_vocab))
    enhanced_model.train(train_tags, train_obs_enhanced)

    enhanced_preds= [enhanced_model.predict(seq) for seq in test_obs_enhanced]

    print(seqeval_report(
        [[INDEX_TO_NER_TAG[t] for t in seq] for seq in test_tags],
        [[INDEX_TO_NER_TAG[t] for t in seq] for seq in enhanced_preds],
        zero_division=0
    ))
    print("Span-Level F1:", f1_score(
        [[INDEX_TO_NER_TAG[t] for t in seq] for seq in test_tags],
        [[INDEX_TO_NER_TAG[t] for t in seq] for seq in enhanced_preds],
        average='micro'
    ))


if __name__== "__main__":
    main()