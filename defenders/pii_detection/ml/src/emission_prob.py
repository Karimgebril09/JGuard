import numpy as np

def find_emission_probabilities(num_states, ner_tags, observed_sequences, vocab_size,vocab_to_index=None):
    emission_probabilities = np.zeros((num_states, vocab_size))
    for i, sequence in enumerate(ner_tags):
        for state, token_idx in zip(sequence, observed_sequences[i]):
            emission_probabilities[state, token_idx] += 1

    row_sums = emission_probabilities.sum(axis=1)
    emission_probabilities = (emission_probabilities + 1) / (row_sums[:, np.newaxis] + emission_probabilities.shape[1])
    return emission_probabilities

