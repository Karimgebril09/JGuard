from pii_inference import PIIDetector
from strategies import PIIStrategy, MaskStrategy


class PIIEngine:
    def __init__(self, detector: PIIDetector, strategy: PIIStrategy = None):
        self.detector = detector
        self.strategy = strategy if strategy is not None else MaskStrategy()

    def set_strategy(self, strategy: PIIStrategy):
        self.strategy = strategy

    def process(self, text: str):
        words, tags = self.detector.predict(text)
        return self.strategy.apply(words, tags)