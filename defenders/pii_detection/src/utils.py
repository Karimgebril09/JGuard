import pandas as pd

def tokenise_with_alignment(tokenizer, text):
    words= text.strip().split()
    subword_ids= []
    word_first_subword = []
    tokens = []

    for word in words:
        pieces = tokenizer.tokenize(word)
        if not pieces:
            pieces = [tokenizer.unk_token]
        ids = tokenizer.convert_tokens_to_ids(pieces)
        tokens.extend(pieces)

        word_first_subword.append(len(subword_ids))
        subword_ids.extend(ids)

    return words, tokens, subword_ids, word_first_subword


def prepare_dataset(df, tokenizer):
    all_sentences = []
    
    for i in range(len(df)):
        words, tokens, _, word_first_subword = tokenise_with_alignment(tokenizer, df["unmasked_text"].iloc[i])
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
        if len(sentence_words) == len(sentence_labels):
            all_sentences.append([sentence_words, sentence_labels])

    return pd.DataFrame(all_sentences, columns=["words", "labels"])
