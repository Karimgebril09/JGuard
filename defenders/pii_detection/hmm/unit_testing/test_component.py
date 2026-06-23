from defenders.pii_detection.hmm.src.viterbi import viterbi
from defenders.pii_detection.hmm.src.components import find_emission_probabilities, find_start_probabilities, find_transition_probabilities
import numpy as np

class TestComponentsFunctions:
    
    def test_start_probabilities_sum_to_one(self):
        
        probs = find_start_probabilities(5, [[0, 1, 2], [1, 2, 3], [3, 4]])
        assert np.isclose(np.sum(probs), 1.0), "Probabilities should sum to 1"
        assert np.all(probs > 0), "No probability should be zero becuse of laplace "
 
    
    def test_transition_probabilities_valid_distribution(self):
        #check if rows sum to 1 and non negative
        trans = find_transition_probabilities(3, [[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        assert np.allclose(np.sum(trans, axis=1), 1.0), "Each row should sum to 1"
        assert np.all(trans >= 0), "All probabilities should be non-negative"
    

    def test_emission_probabilities_no_zero_values(self):
        #check if emission probabilities are non-zero
        emit = find_emission_probabilities(2, [[0, 1]], [[0, 1]], vocab_size=100)
        assert np.all(emit > 0), "No emission probability should be exactly zero"
        

    def test_viterbi_path_length(self):
        obs = [0, 1, 0, 1, 0]
        num_states = 3
        log_start = np.log(np.ones(3) / 3)
        log_trans = np.log(np.ones((3, 3)) / 3)
        log_emit = np.log(np.ones((3, 2)) / 2)
        
        path = viterbi(obs, num_states, log_start, log_trans, log_emit)
        assert len(path) == len(obs) # check length of path matches with word
        assert all(0 <= state < num_states for state in path)   #check if all states are valid indices
    
    def test_viterbi_long_sequence_no_underflow(self):
        
        # Create a 100-token sequence
        obs = [0] * 100
        num_states = 3
        log_start = np.log(np.ones(3) / 3)
        log_trans = np.log(np.ones((3, 3)) / 3)
        log_emit = np.log(np.ones((3, 1)) / 1)
        
        path = viterbi(obs, num_states, log_start, log_trans, log_emit)
        assert len(path) == 100 
        assert np.all(np.isfinite(path))  #should not there any inf
       
        
        
        
if __name__ == "__main__":
    test = TestComponentsFunctions()
    test.test_start_probabilities_sum_to_one()
    test.test_transition_probabilities_valid_distribution()
    test.test_emission_probabilities_no_zero_values()
    test.test_viterbi_path_length()
    test.test_viterbi_long_sequence_no_underflow()