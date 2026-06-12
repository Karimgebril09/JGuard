import hashlib

class PIIStrategy:
    def apply(self, word_tags_pairs):
        raise NotImplementedError


class MaskStrategy(PIIStrategy):
    def apply(self, word_tags_pairs):
        return " ".join(f"<{t[2:]}>" if t != "O" else w for w, t in word_tags_pairs)

class EncryptStrategy(PIIStrategy):
    def _enc(self, x):
        return hashlib.sha256(x.encode()).hexdigest()[:10]

    def apply(self, word_tags_pairs):
        return " ".join(self._enc(w) if t != "O" else w for w, t in word_tags_pairs)


class BlockStrategy(PIIStrategy):
    def apply(self, word_tags_pairs):
        if sum([t != "O" for w, t in word_tags_pairs]) > 0:
            return "[BLOCKED: PII DETECTED]"
        return " ".join(w for w, t in word_tags_pairs)