from defenders.tools.code.src.ast_feature_extractor import extract_security_features_from_ast

def test_ast():
    code1 = """
import os
from subprocess import call

print("Hello, World!")
"""
    _, features = extract_security_features_from_ast(code1)
    assert features["dangerous_imports"] == 2

    code2 = """
import os
def dangerous_function():
    os.system("ls -la")
"""
    _, features = extract_security_features_from_ast(code2)
    assert features["dangerous_imports"] == 1
    assert features["dangerous_calls"] == 1

    # overwriting files
    code3 = """
with open("./defenders/tools/code/test/test.txt", "w") as f:
    f.write("Hello, World!")
"""
    _, features = extract_security_features_from_ast(code3)
    assert features["overwriting_files"] == 1


if __name__ == "__main__":
    test_ast()
    print("All tests passed!")
