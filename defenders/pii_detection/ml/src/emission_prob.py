import numpy as np

def find_emission_probabilities(num_states, ner_tags, observed_sequences, vocab_size):
    emit = np.zeros((num_states, vocab_size))
    for seq_tags, seq_obs in zip(ner_tags, observed_sequences):
        for state, obs in zip(seq_tags, seq_obs):
            if isinstance(obs, list):          
                for idx in obs:
                    emit[state, idx] += 1
            else:                            
                emit[state, obs] += 1
    row_sums = emit.sum(axis=1)
    return (emit + 1) / (row_sums[:, np.newaxis] + vocab_size)
