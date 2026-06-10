
import pickle
from pathlib import Path
from label_maps import INDEX_TO_NER_TAG

class NERPredictor:
    def __init__(self, artifacts_dir: str="./artifacts"):
        base=Path(artifacts_dir)
        with open(base / "hmm_model.pkl", "rb") as f:
            self._model=pickle.load(f)
        with open(base / "vocab_to_index.pkl", "rb") as f:
            self._vocab=pickle.load(f)
        self._unk_idx=self._vocab["UNK"]


    def predict(self, text) :
        tokens=text.strip().split()
        if not tokens:
            return []
        obs=[self._vocab.get(tok, self._unk_idx) for tok in tokens]
        indices=self._model.viterbi_algorithm(obs)
        tags=[INDEX_TO_NER_TAG.get(i, "O") for i in indices]
        return list(zip(tokens, tags))


if __name__=="__main__":
    predictor=NERPredictor()

    test_sentences=[
        "Barack Obama visited Paris last week .",
        "Apple Inc . was founded by Steve Jobs in Cupertino .",
        "The Nile flows through Egypt and Sudan .",
    ]

    for sentence in test_sentences:
        print(f"\nInput : {sentence}")

        pairs=predictor.predict(sentence)
        print("Tokens + Tags:")
        for token, tag in pairs:
            print(f"  {token:<20} {tag}")
