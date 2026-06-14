import hashlib

class PIIStrategy:
    def apply(self, word_tags_pairs):
        raise NotImplementedError


class MaskStrategy(PIIStrategy):
    def apply(self, word_tags_pairs):
        
        final_words = []
        
        for w,t in word_tags_pairs:
            if t != "O":
                final_words.append(f"<{t[2:]}>")
            else:
                final_words.append(w)

        return " ".join(final_words)


class HashStrategy(PIIStrategy):
    def _enc(self, x):
        return hashlib.sha256(x.encode()).hexdigest()[:10]

    def apply(self, word_tags_pairs):
        final_words = []

        for w,t in word_tags_pairs:
            if t != "O":
                final_words.append(self._enc(w))
            else:
                final_words.append(w)
        return " ".join(final_words)


class BlockStrategy(PIIStrategy):
    def apply(self, word_tags_pairs):
        final_words = []

        for w,t in word_tags_pairs:
            if t != "O":
                return "<BLOCKED>"

            final_words.append(w)

        return " ".join(final_words)
    
class PartialMasking(PIIStrategy):
    def apply(self, word_tags_pairs):
        final_words = []

        for w,t in word_tags_pairs:
            if t != "O":
                masked_word = "***" + w[-2:]
                final_words.append(masked_word)
            else:
                final_words.append(w)

        return " ".join(final_words)