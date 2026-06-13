import numpy as np
from pydash import lt
from start_prob import find_start_probabilities
from emission_prob import find_emission_probabilities
from transition_prob import find_transition_probabilities

def viterbi(observations, states, start_probabilities, transition_probabilities, emission_probabilities):
    num_states = len(states)
    num_observations = len(observations)

    log_start = np.log(start_probabilities + 1e-10)
    log_trans  = np.log(transition_probabilities + 1e-10)
    log_emit   = np.log(emission_probabilities + 1e-10)

    viterbi_matrix = np.zeros((num_states, num_observations)) # Initializing the Viterbi matrix and backpointer matrix
    backpointer = np.zeros((num_states, num_observations), dtype=int)
    
    viterbi_matrix[:, 0] = log_start + log_emit[:, observations[0]]
    
    for t in range(1, num_observations):
        for s in range(num_states):
            max_prob = -np.inf
            max_backpointer = -1
            
            # Computing the maximum probability and corresponding backpointer
            for s_prime in range(num_states):
                prob = viterbi_matrix[s_prime, t-1] + log_trans[s_prime, s] + log_emit[s, observations[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_backpointer = s_prime
            
            viterbi_matrix[s, t] = max_prob
            backpointer[s, t] = max_backpointer

    # print ("Viterbi Matrix:\n", viterbi_matrix)
    # print ("Backpointer Matrix:\n", backpointer)
    
    best_path = [-1] * num_observations # We backtrack to find the best state sequence
    best_last_state = np.argmax(viterbi_matrix[:, num_observations - 1])
    best_path[-1] = best_last_state
    
    for t in range(num_observations - 2, -1, -1):
        best_last_state = backpointer[best_last_state, t + 1]
        best_path[t] = best_last_state

    best_path_tags = [states[i] for i in best_path] # Convert the best_path indices to actual NER tags

    return best_path_tags

