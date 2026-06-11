from pii_inference import PIIDetector
from strategies import PIIStrategy, MaskStrategy, EncryptStrategy, BlockStrategy
import warnings
warnings.filterwarnings("ignore")


class PIIEngine:
    def __init__(self, strategy: PIIStrategy = None):
        self.detector = PIIDetector(
            checkpoint_path="./../models/best_bert_bilstm_crf.pt",
            checkpoint_path2="./../models/pii_ner_model.pth"
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
    strategy2 = EncryptStrategy()
    strategy3 = BlockStrategy()


    PIIEngine = PIIEngine(strategy=strategy1)
    result = PIIEngine.process(text)
    print(result)

    PIIEngine.set_strategy(strategy2)
    result = PIIEngine.process(text)
    print(result)

    PIIEngine.set_strategy(strategy3)
    result = PIIEngine.process(text)
    print(result)