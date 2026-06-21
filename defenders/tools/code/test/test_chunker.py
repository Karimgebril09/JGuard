from defenders.tools.code.src.chunker import CodeChunker

def test_chunk_function_based(chunker):
    code = """
import os
print("Hello, World!")

def hello_world():
    print("Hello, World!")

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
    chunks = chunker.chunk_code(code)
    assert len(chunks) == 4


def test2_chunk_function_based(chunker):
    code = """
print("hi from karim")
# this is a code without any function definitions
for i in range(10):
    print(i)
"""
    chunks = chunker.chunk_code(code)
    assert len(chunks) == 1
   

def test_chunk_line_based(chunker):
    code = """import os
print("Hello, World!")
def hello_world():
    print("Hello, World!")
def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
"""
    chunks = chunker.chunk_code(code)
    assert len(chunks) == 2

def test_chunk_line_based_some_remaining_lines(chunker):
    code = """import os
print("hellllllooooooo")
def hello_world():
    print("Hello, World!")
print("this is last print in this code")
"""
    chunks = chunker.chunk_code(code)
    assert len(chunks) == 2

def run_tests():
    chunker_function = CodeChunker(strategy="function")
    test_chunk_function_based(chunker_function)
    test2_chunk_function_based(chunker_function)

    chunker_line = CodeChunker(strategy="line", chunk_size=4)
    test_chunk_line_based(chunker_line)
    chunker_line = CodeChunker(strategy="line", chunk_size=4)
    test_chunk_line_based_some_remaining_lines(chunker_line)

    print("All tests passed!")


if __name__ == "__main__":
    run_tests()