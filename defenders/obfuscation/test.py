import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from defenders.obfuscation.pipeline import run_obfuscation_pipeline as run_obfuscation

result = run_obfuscation("my name is <EMAIL>")
print("canonical:", result["clean_text"])
print("metadata:", result["metadata_envelope"])
print("state outputs:", result["stage_outputs"])