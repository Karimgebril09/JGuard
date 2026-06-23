from defenders.pii_detection.src.strategies import PIIStrategy, MaskStrategy, HashStrategy, BlockStrategy, PartialMasking
import hashlib


def test_mask_strategy():
    strategy = MaskStrategy()
    input_data = [("My", "O"), ("email", "O"), ("is", "O"), ("johndoe@email.com", "B-EMAIL")]
    expected_output = "My email is <EMAIL>"
    assert strategy.apply(input_data) == expected_output, "MaskStrategy test failed"

def test_hash_strategy():
    strategy = HashStrategy()
    input_data = [("My", "O"), ("email", "O"), ("is", "O"), ("johndoe@email.com", "B-EMAIL")]

    expected_output = f"My email is {hashlib.sha256('johndoe@email.com'.encode()).hexdigest()[:10]}"
    assert strategy.apply(input_data) == expected_output, "HashStrategy test failed"    

def test_block_strategy():
    strategy = BlockStrategy()
    input_data = [("My", "O"), ("email", "O"), ("is", "O"), ("johndoe@email.com", "B-EMAIL")]
    expected_output = "<BLOCKED>"
    assert strategy.apply(input_data) == expected_output, "BlockStrategy test failed"

def test_partial_masking_strategy():
    strategy = PartialMasking()
    input_data = [("My", "O"), ("email", "O"), ("is", "O"), ("johndoe@email.com", "B-EMAIL")]
    expected_output = "My email is ***om"
    assert strategy.apply(input_data) == expected_output, "PartialMasking test failed"


def run_tests():
    test_mask_strategy()
    test_hash_strategy()
    test_block_strategy()
    test_partial_masking_strategy()
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()