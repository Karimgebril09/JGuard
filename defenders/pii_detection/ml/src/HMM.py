from viterbi import viterbi
import numpy as np
from start_prob import find_start_probabilities
from transition_prob import find_transition_probabilities
from emission_prob import find_emission_probabilities

class HMM:
    def __init__(self, num_states, vocab_size):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.log_start = None
        self.log_trans = None
        self.log_emit  = None

    def train(self, tag_sequences, observed_sequences):
        self.log_start = np.log(find_start_probabilities(self.num_states, tag_sequences) + 1e-10)
        self.log_trans = np.log(find_transition_probabilities(self.num_states, tag_sequences) + 1e-10)
        emit = find_emission_probabilities(self.num_states, tag_sequences, observed_sequences, self.vocab_size)
        self.log_emit  = np.log(emit + 1e-10)

    def predict(self, observed_sequence):
        return viterbi(observed_sequence, self.num_states, self.log_start, self.log_trans, self.log_emit)