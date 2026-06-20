import re


class Chunk:
    def __init__(self, chunk_id, text, source, start_word, end_word):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.start_word = start_word
        self.end_word = end_word


class DocumentChunker:

    def __init__(self, chunk_size=300, overlap=50):
        if overlap >= chunk_size:
            
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_file(self, filename):
        file = open(filename, "r", encoding="utf-8")
        text = file.read()
        file.close()

        return self.chunk_text(text, filename)

    def load_text(self, text):
        return self.chunk_text(text, "input_text")

    def chunk_text(self, text, source):

        text = re.sub(r"\s+", " ", text)

        words = text.split()

        chunks = []

        start = 0
        chunk_number = 0

        # print("Chunking:", source)

        while start < len(words):

            end = start + self.chunk_size

            if end > len(words):
                end = len(words)

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)
            chunk = Chunk(chunk_id=str(chunk_number),text=chunk_text,source=source,start_word=start,end_word=end)
            chunks.append(chunk)

            # print("Created chunk", chunk_number,
            #       "from word", start,
            #       "to", end)

            chunk_number += 1
            start = start + (self.chunk_size - self.overlap)

        return chunks