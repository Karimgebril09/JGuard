from sklearn.base import defaultdict


def threshold_data(sequences, threshold):
    word_counts = defaultdict(int)
    for sequence in sequences:
        for token, state in sequence:
            word_counts[token] += 1

    unk_sequences = []
    for sequence in sequences:
        unk_sequence = [(token if word_counts[token] > threshold else 'UNK', state) for token, state in sequence]
        unk_sequences.append(unk_sequence)

    return unk_sequences


