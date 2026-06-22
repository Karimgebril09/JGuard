import re

class CodeChunker:
    def __init__(self, strategy="line", chunk_size=1000):
        self.strategy = strategy
        self.chunk_size = chunk_size
        
    def function_based_chunking(self, code) -> list:
        pattern = r"def\s+\w+\s*\(.*?\):"
        matches = list(re.finditer(pattern, code))

        chunks = []

        initial_start = 0
        end = matches[0].start() if matches else len(code)
        if initial_start < end:
            chunks.append(code[initial_start:end])

        
        for i, match in enumerate(matches):
            start = match.start()
            if i+1 < len(matches):
                end= matches[i + 1].start()
            else:
                end=len(code)

            chunks.append(code[start:end])
       
        return chunks
    

    def line_based_chunking(self, code) -> list:
        lines = code.splitlines()
        chunks = []
        current_chunk = []
        for line in lines:
            current_chunk.append(line)
            if len(current_chunk) >= self.chunk_size:
                chunk="\n".join(current_chunk)
                chunks.append(chunk)
                current_chunk = []

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks
    

    def chunk_code(self, code) -> list:
        if self.strategy == "function":
            return self.function_based_chunking(code)
        elif self.strategy == "line":
            return self.line_based_chunking(code)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")