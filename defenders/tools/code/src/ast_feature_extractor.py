import ast
import os
# some predefined imports and calls/functions
DANGEROUS_IMPORTS = ["os", "subprocess", "socket", "ctypes","requests", "paramiko", "shutil","base64","sys"]
DANGEROUS_FUNCS = ["eval", "exec", "compile", "__import__"]

def extract_security_features_from_ast(code):
    features = {"dangerous_imports": 0,"dangerous_calls": 0,"overwriting_files": 0}
    try:
        tree = ast.parse(code)
    except:
        # failed to parse the code might be syntax error
        return False, features

    for node in ast.walk(tree):
        try:
            # handling import 
            if type(node) == ast.Import:
                for name in node.names:
                    if name.name.split(".")[0] in DANGEROUS_IMPORTS:
                        features["dangerous_imports"] += 1

            # handling from import case
            if type(node) == ast.ImportFrom:
                import_name=node.module or ""
                if import_name.split(".")[0] in DANGEROUS_IMPORTS:
                    features["dangerous_imports"] += 1

            # handling function calls
            if type(node) == ast.Call:
                # case it call directly the function 
                if hasattr(node.func, "id"):
                    if node.func.id in DANGEROUS_FUNCS:
                        features["dangerous_calls"] += 1

                # case it combine library with call like os.system
                if hasattr(node.func, "attr"):
                    if node.func.attr == "system":
                        features["dangerous_calls"] += 1

                # overwriting files detection
                if hasattr(node.func, "id") and node.func.id == "open":
                    if len(node.args) > 0:
                        filename = node.args[0].value # gets the file name
                        operation = node.args[1].value if len(node.args) > 1 else None
                        if operation!=None and operation == "w" and os.path.exists(filename): # check if exist actually
                            features["overwriting_files"] += 1

        except Exception as e:
            continue

     
    return True,features