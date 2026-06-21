import re


class Chunk:
    """Represents a chunk of text with metadata."""
    def __init__(self, chunk_id, text, source, start_word, end_word):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.start_word = start_word
        self.end_word = end_word


class DocumentChunker:
    """responsible for chunking and loading text documents into chunks"""
    def __init__(self, chunk_size=300, overlap=50):
        if overlap >= chunk_size:
            self.overlap = chunk_size - 1       
            raise ValueError("overlap must be smaller than chunk_size")
        else :   
            self.overlap = overlap
        self.chunk_size = chunk_size

    def load_file(self, filename):
        """Load a text file and chunk its content"""
        file = open(filename, "r", encoding="utf-8")
        text = file.read()
        file.close()

        return self.chunk_text(text, filename)

    def load_text(self, text):
        """load text directly """
        return self.chunk_text(text, "input_text")

    def chunk_text(self, text, source):

        text = re.sub(r"\s+", " ", text) # remove extra whitespace

        words = text.split() 

        chunks = []

        start = 0
        chunk_number = 0

        while start < len(words):

            end = start + self.chunk_size

            if end > len(words): # make sure that dont exceed the length of the words
                end = len(words)

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)
            chunk = Chunk(chunk_id=str(chunk_number),text=chunk_text,source=source,start_word=start,end_word=end)
            chunks.append(chunk)  # create chunk and add it to the list 

            chunk_number += 1
            start = start + (self.chunk_size - self.overlap)

        return chunks