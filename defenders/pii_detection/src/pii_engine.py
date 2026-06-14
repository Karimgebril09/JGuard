import os,sys

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from defenders.pii_detection.src.pii_inference import PIIDetector
from defenders.pii_detection.src.strategies import PIIStrategy, MaskStrategy, HashStrategy, BlockStrategy,PartialMasking

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "..", "models")


class PIIEngine:
    def __init__(self, strategy: PIIStrategy | None = None):
        self.detector = PIIDetector(
            checkpoint_path=os.path.join(_MODELS_DIR, "best_bert_bilstm_crf.pt"),
            checkpoint_path2=os.path.join(_MODELS_DIR, "pii_ner_model.pth")
        )
        self.strategy = strategy if strategy is not None else MaskStrategy()

    def set_strategy(self, strategy: PIIStrategy):
        self.strategy = strategy

    def process(self, text: str):
        word_tags_pairs = self.detector.predict(text)
        return self.strategy.apply(word_tags_pairs)
    


if __name__ == "__main__":
    # test the engine with all strategies
    text = "My email is john.doe@example.com"

    strategy1 = MaskStrategy()
    strategy2 = HashStrategy()
    strategy3 = BlockStrategy()
    strategy4 = PartialMasking()


    pii_engine = PIIEngine(strategy=strategy1)
    result = pii_engine.process(text)
    print(result)

    pii_engine.set_strategy(strategy2)
    result = pii_engine.process(text)
    print(result)

    pii_engine.set_strategy(strategy3)
    result = pii_engine.process(text)
    print(result)


    pii_engine.set_strategy(strategy4)
    result = pii_engine.process(text)
    print(result)