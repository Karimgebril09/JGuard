import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model_interface import classify_text, classify_text_with_details


def main() -> None:
    input_text = "hi everything is alright"

    result = classify_text(input_text)

    print(result)

    # print("Input:")
    # print(input_text)
    # print("\nPrediction:")
    # print(f"  label: {result['label']} ({result['class_name']})")
    # print(f"  probability: {result['probability']:.6f}")
    # print(f"  threshold: {result['threshold']:.6f}")


if __name__ == "__main__":
    main()
