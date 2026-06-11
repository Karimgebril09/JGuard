import numpy as np

def find_transition_probabilities(number_of_states, ner_tags):
    #init zeros matrix
    trans_prob = np.zeros((number_of_states, number_of_states))

    #fill freq
    for sequence in ner_tags:
        for i in range(len(sequence) - 1):
            from_s = sequence[i]
            to_s = sequence[i + 1]
            trans_prob[from_s, to_s] += 1
    #divide by freq of row (from state)
    row_sums = trans_prob.sum(axis=1)
    trans_prob = (trans_prob + 1) / (row_sums[:, np.newaxis] + number_of_states)

    return trans_prob
