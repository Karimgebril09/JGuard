"""
inference.py
------------
HMM NER inference as a reusable class.

Usage from anywhere:
    from inference import NERPredictor

    predictor = NERPredictor()                     # loads from ./artifacts by default
    predictor = NERPredictor("path/to/artifacts")  # custom path

    predictor.predict("Barack Obama visited Paris")
    # → [("Barack", "B-PER"), ("Obama", "I-PER"), ("visited", "O"), ("Paris", "B-LOC")]

    predictor.predict_tags("Barack Obama visited Paris")
    # → ["B-PER", "I-PER", "O", "B-LOC"]

    predictor.predict_entities("Barack Obama visited Paris")
    # → [{"text": "Barack Obama", "label": "PER", "tokens": [...], "start": 0, "end": 1}]
"""

import pickle
from pathlib import Path

from label_maps import INDEX_TO_NER_TAG


class NERPredictor:
    def __init__(self, artifacts_dir: str = "./artifacts"):
        base = Path(artifacts_dir)
        with open(base / "hmm_model.pkl", "rb") as f:
            self._model = pickle.load(f)
        with open(base / "vocab_to_index.pkl", "rb") as f:
            self._vocab = pickle.load(f)
        self._unk_idx = self._vocab["UNK"]


    def predict(self, text: str) -> list[tuple[str, str]]:
        """
        Returns a list of (token, tag) pairs.

        Example:
            [("Barack", "B-PER"), ("Obama", "I-PER"), ("visited", "O"), ("Paris", "B-LOC")]
        """
        tokens = text.strip().split()
        if not tokens:
            return []
        obs     = [self._vocab.get(tok, self._unk_idx) for tok in tokens]
        indices = self._model.viterbi_algorithm(obs)
        tags    = [INDEX_TO_NER_TAG.get(i, "O") for i in indices]
        return list(zip(tokens, tags))

    def predict_tags(self, text: str) -> list[str]:
        """Returns only the tag sequence (same order as tokens)."""
        return [tag for _, tag in self.predict(text)]

    def predict_entities(self, text: str) -> list[dict]:
        """
        Returns a list of entity dicts, merging consecutive B-/I- spans.

        Each dict:
            {
                "text":   str,        # full entity surface form
                "label":  str,        # entity type, e.g. "PER", "LOC"
                "tokens": list[str],  # individual tokens in the span
                "start":  int,        # token index of span start (inclusive)
                "end":    int,        # token index of span end   (inclusive)
            }
        """
        pairs = self.predict(text)
        entities, current = [], None

        for i, (token, tag) in enumerate(pairs):
            if tag.startswith("B-"):
                if current:
                    entities.append(current)
                label = tag[2:]
                current = {"text": token, "label": label,
                           "tokens": [token], "start": i, "end": i}

            elif tag.startswith("I-") and current and tag[2:] == current["label"]:
                current["tokens"].append(token)
                current["text"] += f" {token}"
                current["end"] = i

            else:
                if current:
                    entities.append(current)
                current = None

        if current:
            entities.append(current)

        return entities


if __name__ == "__main__":
    predictor = NERPredictor()

    test_sentences = [
        "Barack Obama visited Paris last week .",
        "Apple Inc . was founded by Steve Jobs in Cupertino .",
        "The Nile flows through Egypt and Sudan .",
    ]

    for sentence in test_sentences:
        print(f"\nInput : {sentence}")

        pairs = predictor.predict(sentence)
        print("Tokens + Tags:")
        for token, tag in pairs:
            print(f"  {token:<20} {tag}")

        entities = predictor.predict_entities(sentence)
        print("Entities:")
        if entities:
            for ent in entities:
                print(f"  [{ent['label']}] '{ent['text']}'  (tokens {ent['start']}–{ent['end']})")
        else:
            print("  (none found)")