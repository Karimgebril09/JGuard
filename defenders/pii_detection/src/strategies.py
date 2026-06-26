import hashlib

class PIIStrategy:
    def apply(self, word_tags_pairs):
        raise NotImplementedError

class MaskStrategy(PIIStrategy):
    def apply(self, word_tags_pairs):
        
        final_words = []
        
        for w,t in word_tags_pairs:
            if t != "O":
                final_words.append(f"<{t[2:]}>") # skips first two tags of B- or I-
            else:
                final_words.append(w)

        return " ".join(final_words)


class HashStrategy(PIIStrategy):
    def _hash(self, x):
        hashed_value = hashlib.sha256(x.encode()).hexdigest()
        return hashed_value[:10] # get first 10 characters of the hash

    def apply(self, word_tags_pairs):
        final_words = []

        for w,t in word_tags_pairs:
            if t != "O":
                final_words.append(self._hash(w))
            else:
                final_words.append(w)
        return " ".join(final_words)


class BlockStrategy(PIIStrategy):
    def apply(self, word_tags_pairs):
        final_words = []

        for w,t in word_tags_pairs:
            if t != "O": # if any non "o" tag is found we return blocked immediatly
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