import hashlib

class PIIStrategy:
    def apply(self, words, tags):
        raise NotImplementedError


class MaskStrategy(PIIStrategy):
    def apply(self, words, tags):
        return " ".join(f"<{t[2:]}>" if t != "O" else w for w, t in zip(words, tags))

class EncryptStrategy(PIIStrategy):
    def _enc(self, x):
        return hashlib.sha256(x.encode()).hexdigest()[:10]

    def apply(self, words, tags):
        return " ".join(self._enc(w) if t != "O" else w for w, t in zip(words, tags))


class BlockStrategy(PIIStrategy):
    def apply(self, words, tags):
        if sum([t != "O" for t in tags]) > 0:
            return "[BLOCKED: PII DETECTED]"
        return " ".join(words)