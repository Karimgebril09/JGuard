from defenders.pii_detection.src.pii_inference import PIIDetector
import os
_HERE = os.path.dirname(os.path.abspath(__file__))

def test_voting_mechanism(detector):
    predictions1 = ["O", "B-EMAIL", "O", "O"]
    predictions2 = ["O", "B-EMAIL", "O", "O"]
    predictions3 = ["O", "O", "O", "O"]
    assert detector.voting_strategy(predictions1, predictions2, predictions3) == ["O", "B-EMAIL", "O", "O"], "Voting mechanism test failed"


def test_trusted_tags(detector):
    predictions1 = ["B-IPv4", "B-IPv6", "O", "O"]
    predictions2 = ["B-IPv6", "O", "O", "O"]
    predictions3 = ["O", "O", "O", "O"]
    assert detector.trust_strategy( predictions1, 
                                    predictions2, 
                                    predictions3) == ["B-IPv4", "B-IPv6", "O", "O"], "Trusted tags test failed"
    
    predictions1 = ["O", "O", "O", "O"]
    predictions2 = ["O", "O", "O", "O"]
    predictions3 = ["B-EMAIL", "O", "O", "O"]
    assert detector.trust_strategy( predictions1,
                                   predictions2,
                                   predictions3) == ["B-EMAIL", "O", "O", "O"], "Trusted tags test failed"
    

def run_tests():
    checkpoint_path = os.path.join(_HERE, "..", "models", "distilbert_bilstm_crf.pt")
    checkpoint_path2 = os.path.join(_HERE, "..", "models", "pii_ner_model.pth")
    detector = PIIDetector(checkpoint_path=checkpoint_path, checkpoint_path2=checkpoint_path2)
    test_voting_mechanism(detector)
    test_trusted_tags(detector)
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()

