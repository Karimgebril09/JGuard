import numpy as np

def find_emission_probabilities(num_states, ner_tags, observed_sequences, vocab_size):

    emission = np.zeros((num_states, vocab_size))

    for tags, obs_seq in zip(ner_tags, observed_sequences):
        for state, obs in zip(tags, obs_seq):
            if isinstance(obs, list):
                for word in obs:
                    emission[state][word] += 1
            else:
                emission[state][obs] += 1

    # Laplace smoothing so if word not in train data not get 0
    for state in range(num_states):
        total = 0
        for word in range(vocab_size):
            total += emission[state][word]

        for word in range(vocab_size):
            emission[state][word] = (emission[state][word] + 1) / (total + vocab_size)

    return emission


def find_start_probabilities(num_states, sequences_tagged):
    counts = np.zeros(num_states)

    # Count starting states
    for seq in sequences_tagged:
        if len(seq) > 0:
            counts[seq[0]] += 1

    start_probs = np.zeros(num_states)
    for state in range(num_states):
        start_probs[state] = (counts[state] + 1) / (len(sequences_tagged) + num_states)

    return start_probs
def find_transition_probabilities(num_states, sequences_tagged):

    trans = np.zeros((num_states, num_states))
    for seq in sequences_tagged:

        for i in range(len(seq) - 1):
            current_s = seq[i]
            next_s = seq[i + 1]
            trans[current_s][next_s] += 1

    # Laplace smoothing so if word not in train data not get 0
    for state in range(num_states):
        total = 0
        for next_state in range(num_states):
            total += trans[state][next_state]
        for next_state in range(num_states):
            trans[state][next_state] = (trans[state][next_state] + 1) / (total + num_states)

    return trans