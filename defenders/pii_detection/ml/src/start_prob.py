import numpy as np

def find_start_probabilities(num_states, sequences_tagged):
    start_states = [seq[0] for seq in sequences_tagged if len(seq) > 0]
    counts = np.bincount(start_states, minlength=num_states).astype(float)
    return (counts + 1) / (len(sequences_tagged) + num_states)  

