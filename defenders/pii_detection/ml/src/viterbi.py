import numpy as np

# def viterbi(observations, states, start_probabilities, transition_probabilities, emission_probabilities):
#     num_states = len(states)
#     num_observations = len(observations)

#     log_start = np.log(start_probabilities + 1e-10)
#     log_trans  = np.log(transition_probabilities + 1e-10)
#     log_emit   = np.log(emission_probabilities + 1e-10)

#     viterbi_matrix = np.zeros((num_states, num_observations)) # Initializing the Viterbi matrix and backpointer matrix
#     backpointer = np.zeros((num_states, num_observations), dtype=int)
    
#     viterbi_matrix[:, 0] = log_start + np.array([emit_score(log_emit, s, observations[0]) for s in range(num_states)])
    
#     for t in range(1, num_observations):
#         for s in range(num_states):
#             max_prob = -np.inf
#             max_backpointer = -1
            
#             # Computing the maximum probability and corresponding backpointer
#             for s_prime in range(num_states):
#                 prob = viterbi_matrix[s_prime, t-1] + log_trans[s_prime, s] + emit_score(log_emit, s, observations[t])  
#                 if prob > max_prob:
#                     max_prob = prob
#                     max_backpointer = s_prime
            
#             viterbi_matrix[s, t] = max_prob
#             backpointer[s, t] = max_backpointer

#     # print ("Viterbi Matrix:\n", viterbi_matrix)
#     # print ("Backpointer Matrix:\n", backpointer)
    
#     best_path = [-1] * num_observations #
#     best_last_state = np.argmax(viterbi_matrix[:, num_observations - 1])
#     best_path[-1] = best_last_state
    
#     for t in range(num_observations - 2, -1, -1):
#         best_last_state = backpointer[best_last_state, t + 1]
#         best_path[t] = best_last_state


#     return best_path 

def emit_score(log_emit, state, feature_indices):
    return sum(log_emit[state, fi] for fi in feature_indices)


def viterbi(observations, num_states, log_start, log_trans, log_emit):
    T = len(observations)
    dp  = np.zeros((num_states, T))
    bp  = np.zeros((num_states, T), dtype=int)

    def emit_score(s, obs):
        if isinstance(obs, list):
            return sum(log_emit[s, idx] for idx in obs)
        return log_emit[s, obs]

    dp[:, 0] = log_start + np.array([emit_score(s, observations[0]) for s in range(num_states)])

    for t in range(1, T):
        scores = dp[:, t - 1][:, None] + log_trans        # (num_states, num_states)
        bp[:, t] = scores.argmax(axis=0)
        dp[:, t] = scores.max(axis=0) + np.array([emit_score(s, observations[t]) for s in range(num_states)])

    best_path = [0] * T
    best_path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(T - 2, -1, -1):
        best_path[t] = bp[best_path[t + 1], t + 1]
    return best_path

