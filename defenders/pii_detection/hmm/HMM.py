from viterbi import viterbi
import numpy as np
from components import find_start_probabilities, find_transition_probabilities, find_emission_probabilities

class HMM:
    def __init__(self, num_states, vocab_size):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.start = None
        self.trans = None
        self.emit  = None

    def train(self, tag_sequences, observed_sequences):
        self.start = find_start_probabilities(self.num_states, tag_sequences)   
        self.trans = find_transition_probabilities(self.num_states, tag_sequences)
        self.emit  = find_emission_probabilities(self.num_states, tag_sequences, observed_sequences, self.vocab_size)

    def predict(self, observed_sequence):
        return viterbi(observed_sequence, self.num_states, self.start, self.trans, self.emit)