
from defenders.pii_detection.hmm.src.HMM import HMM


class TestHMMModel:
    
    def test_hmm_train_updates_probabilities(self):
        
        model = HMM(3, 20)
        tag_seqs = [[0, 1, 2], [1, 2, 0]]
        obs_seqs = [[0, 1, 2], [3, 4, 5]]
        
        model.train(tag_seqs, obs_seqs)
        # all correctly trianed
        assert model.start is not None
        assert model.trans is not None
        assert model.emit is not None
    
    def test_hmm_predict_basic(self):
        model = HMM(2, 10)
        model.train([[0, 1], [1, 0]], [[0, 1], [2, 3]])
        
        prediction = model.predict([0, 1])
        assert len(prediction) == 2
        assert all(p in [0, 1] for p in prediction)
    
    def test_hmm_predict_single_token(self):
        
        model = HMM(3, 15)
        model.train([[0], [1], [2]], [[0], [5], [10]])
        #check predict funtion 
        prediction = model.predict([[5]])
        assert len(prediction) == 1

if __name__ == "__main__":
    test = TestHMMModel()
    test.test_hmm_train_updates_probabilities()
    test.test_hmm_predict_basic()
    test.test_hmm_predict_single_token()