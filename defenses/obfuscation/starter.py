"""Entry point for the obfuscation defense pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:
    from .stage1_profiling import RawInputProfile, profile_input
    from .stage2_decoding import DecodedInput, DecodingStep, decode_stage2
    from .stage3_unicoding import Stage3NormalizedInput, normalize_stage3
    from .stage4_leet import Stage4ResolvedInput, resolve_stage4
    from .stage5_defragmenting import Stage5DefragmentedInput, defragment_stage5
    from .stage6_canonicalizing import Stage6CanonicalInput, canonicalize_stage6
    from .stage7_metadata import (
        AnomalyRecord,
        Stage7ProvenanceEnvelope,
        TransformationRecord,
        package_stage7,
    )
    from .pipeline import PipelineResult, run_obfuscation_pipeline
except ImportError:  # pragma: no cover - fallback for direct execution
    from stage1_profiling import RawInputProfile, profile_input
    from stage2_decoding import DecodedInput, DecodingStep, decode_stage2
    from stage3_unicoding import Stage3NormalizedInput, normalize_stage3
    from stage4_leet import Stage4ResolvedInput, resolve_stage4
    from stage5_defragmenting import Stage5DefragmentedInput, defragment_stage5
    from stage6_canonicalizing import Stage6CanonicalInput, canonicalize_stage6
    from stage7_metadata import (
        AnomalyRecord,
        Stage7ProvenanceEnvelope,
        TransformationRecord,
        package_stage7,
    )
    from pipeline import PipelineResult, run_obfuscation_pipeline

__all__ = [
    "RawInputProfile",
    "profile_input",
    "DecodedInput",
    "DecodingStep",
    "decode_stage2",
    "Stage3NormalizedInput",
    "normalize_stage3",
    "Stage4ResolvedInput",
    "resolve_stage4",
    "Stage5DefragmentedInput",
    "defragment_stage5",
    "Stage6CanonicalInput",
    "canonicalize_stage6",
    "TransformationRecord",
    "AnomalyRecord",
    "Stage7ProvenanceEnvelope",
    "package_stage7",
    "PipelineResult",
    "run_obfuscation_pipeline",
]


def main(argv: list[str] | None = None) -> int:
    """Run stage-1 profiling from the command line."""

    parser = argparse.ArgumentParser(description="Run obfuscation defense stage 1 profiler.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", type=str, help="Raw input text to profile.")
    source_group.add_argument("--file", type=Path, help="Path to a text file to profile.")
    parser.add_argument("--encoding", default="utf-8", help="Encoding for text file input.")
    parser.add_argument(
        "--as-bytes",
        action="store_true",
        help="Profile bytes directly instead of decoded text.",
    )

    args = parser.parse_args(argv)

    if args.file is not None:
        if args.as_bytes:
            raw_input: str | bytes = args.file.read_bytes()
        else:
            raw_input = args.file.read_text(encoding=args.encoding)
    else:
        raw_input = args.text.encode(args.encoding, errors="replace") if args.as_bytes else args.text

    profile = profile_input(raw_input, encoding=args.encoding)
    print(json.dumps(asdict(profile), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

