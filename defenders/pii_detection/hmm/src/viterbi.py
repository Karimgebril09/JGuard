import numpy as np

def get_emission_score(log_emit,state,observation):
    score= 0
    if isinstance(observation,list):
        # multiple features
        for feature in observation:
            score += log_emit[state][feature]
    else:
        # single feature 
        score= log_emit[state][observation]
    return score

def viterbi(observations,num_states,log_start, log_trans, log_emit):
    T= len(observations)
    dp= np.zeros((num_states, T))
    backpointer= np.zeros((num_states,T), dtype=int)

    # Initialize first column
    for state in range(num_states):
        dp[state][0]= ( log_start[state] +get_emission_score(log_emit, state,observations[0]))

    # Fill DP table
    for t in range(1, T):

        for current_state in range(num_states):
            
            best_score= float("-inf")
            best_prev_state= 0

            for prev_state in range(num_states):
                # max of prevtag * transiton 
                score= (dp[prev_state][t - 1]+log_trans[prev_state][current_state])
                if score > best_score:
                    best_score= score
                    best_prev_state= prev_state
            # multiply by emission score
            dp[current_state][t]= (best_score + get_emission_score(log_emit,current_state, observations[t]   ) )
            # save that at this state i came form the best_prev_state
            backpointer[current_state][t]= best_prev_state

    # Find final best state
    best_last_state= 0
    best_score= dp[0][T - 1]

    for state in range(1, num_states):
        # find the best score at the last time step
        if dp[state][T - 1] >best_score:
            best_score= dp[state][T - 1]
            best_last_state= state

    # Backtrack
    best_path= [0] *T
    best_path[T - 1]=best_last_state

    for t in range(T -2,-1,-1):
        best_path[t]= backpointer[best_path[t +1]][t+ 1]

    return best_path




def viterbi_beam(observations,num_states,log_start,log_trans, log_emit,beam_width=5):
   
   #use beam search to keep the top k only this better for large space 
    T= len(observations)
    dp= np.full((num_states, T), float("-inf"))
    backpointer= np.zeros((num_states, T), dtype=int)
    
    # Initialize
    for state in range(num_states):
        dp[state][0]= log_start[state]+ get_emission_score(log_emit,state, observations[0])
    
    # Forward pass with beam
    for t in range(1, T):
        # Get top beam_width states from previous timestep
        top_prev_states= np.argsort(dp[:, t- 1])[-beam_width:]
        
        for current_state in range(num_states):
            best_score= float("-inf")
            best_prev_state= 0
            
            # Only consider top states (beam constraint)
            for prev_state in top_prev_states:
                score= dp[prev_state, t- 1] + log_trans[prev_state][current_state]
                if score > best_score:
                    best_score= score
                    best_prev_state= prev_state
            
            if best_score > float("-inf"):
                dp[current_state][t]= best_score+get_emission_score(log_emit, current_state, observations[t])
                backpointer[current_state][t]= best_prev_state
    
    # Backtrack
    best_last_state= np.argmax(dp[:,T-1])
    path= [0]* T
    path[T -1]= best_last_state
    
    for t in range(T-2,-1, -1):
        path[t]= backpointer[path[t +1]][t+ 1]
    
    return path
 