from viterbi import viterbi

from start_prob import find_start_probabilities
from transition_prob import find_transition_probabilities
from emission_prob import find_emission_probabilities

class HMM:
    def __init__(self, states, vocab_to_index):
        self.states = states
        self.vocab_to_index = vocab_to_index

        self.start_probabilities = None
        self.transition_probabilities = None
        self.emission_probabilities = None
        self.train_observed_sequences = None


    def compute_start_probabilities(self, ner_tags):
        self.start_probabilities = find_start_probabilities(len(self.states), ner_tags)


    def compute_transition_probabilities(self, ner_tags):
        self.transition_probabilities = find_transition_probabilities(len(self.states), ner_tags)


    def compute_emission_probabilities(self, ner_tags, observed_sequences):
        self.emission_probabilities = find_emission_probabilities(len(self.states), ner_tags, observed_sequences,vocab_size=len(self.vocab_to_index),vocab_to_index=self.vocab_to_index)


    def viterbi_algorithm(self, obs):
        return viterbi(obs, self.states, self.start_probabilities, self.transition_probabilities, self.emission_probabilities)
    
    def train_hmm(self, taged_sequences, observed_sequences):
    
        self.compute_start_probabilities(taged_sequences)
        self.compute_transition_probabilities(taged_sequences)
        self.compute_emission_probabilities(taged_sequences, observed_sequences)
        
    