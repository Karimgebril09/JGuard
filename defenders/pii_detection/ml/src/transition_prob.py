import numpy as np

def find_transition_probabilities(num_states, sequences_tagged):
    trans = np.zeros((num_states, num_states))
    for seq in sequences_tagged:
        for i in range(len(seq) - 1):
            trans[seq[i], seq[i + 1]] += 1
    row_sums = trans.sum(axis=1)
    return (trans + 1) / (row_sums[:, np.newaxis] + num_states)