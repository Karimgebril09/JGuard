import numpy as np

def find_start_probabilities(num_states, sequences_taged):


    start_states = [seq[0] for seq in sequences_taged if len(seq) > 0]
    start_state_counts = np.bincount(start_states, minlength=num_states)
    # print (start_state_counts)
    
    start_probabilities = start_state_counts / len(sequences_taged)
    

    return start_probabilities
