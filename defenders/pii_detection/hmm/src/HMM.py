import joblib

from defenders.pii_detection.hmm.src.viterbi import viterbi ,viterbi_beam
import numpy as np
from defenders.pii_detection.hmm.src.components import find_start_probabilities, find_transition_probabilities, find_emission_probabilities

class HMM:
    def __init__(self, num_states, vocab_size):
        # keep meta data
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.start = None
        self.trans = None
        self.emit  = None
    #when train call theem to get metrics 
    
    def train(self, tag_sequences, observed_sequences):
        self.start = self.safe_log(find_start_probabilities(self.num_states, tag_sequences))
        self.trans = self.safe_log(find_transition_probabilities(self.num_states, tag_sequences))
        self.emit  = self.safe_log(find_emission_probabilities(self.num_states, tag_sequences, observed_sequences, self.vocab_size))
    #when predict use the viterbi
    def predict(self, observed_sequence):
        return viterbi(observed_sequence, self.num_states, self.start, self.trans, self.emit)
    
    def save(self, filepath):
        
        data = {
            'start': self.start,
            'trans': self.trans,
            'emit': self.emit,
            'num_states': self.num_states,
            'vocab_size': self.vocab_size,
            'name': "hmm_model",
           
        }
        joblib.dump(data, filepath)
        
    def safe_log(self, x, epsilon=1e-10):
           # prevnet from log(0)
            return np.log(np.clip(x, epsilon, 1.0))
    def load(self, cls, filepath):
        
        #read model from file 
        try:
            data = joblib.load(filepath)
            model = cls(data['num_states'], data['vocab_size'], data['name'])
            model.start = data['start']
            model.trans = data['trans']
            model.emit = data['emit']
       
            return model
        except Exception as e:
            print (f"Error loading model from {filepath}: {e}")
            raise