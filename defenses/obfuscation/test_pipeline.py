import csv
from pathlib import Path
import sys

from defenses.obfuscation.pipeline import run_obfuscation_pipeline

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(errors="replace")
if callable(stderr_reconfigure):
    stderr_reconfigure(errors="replace")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    cases_path = base_dir / "obfuscation_test_cases.csv"

    with cases_path.open("r", encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle))

    failed = 0
    errored = 0

    for case in cases:
        raw_input = case["input"]
        expected = case["expected_output"]

        try:
            result = run_obfuscation_pipeline(raw_input)
            actual = result["stage_outputs"]["stage6"]["canonical_text"]
        except Exception as exc:  # pragma: no cover - runtime visibility only
            errored += 1
            print(f"\ninput: {raw_input}")
            print(f"output: ERROR: {exc}\n")
            continue

        print(f"\ninput: {raw_input}")
        print(f"output: {actual}\nexpected: {expected}\n")

        if actual != expected:
            failed += 1

    return 0 if failed == 0 and errored == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
