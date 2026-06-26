from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .attacks import (
    ATTACK_TYPES,
    plugin_to_attack_type_map,
)
from .config import (
    DATASET_PATH,
    DEFAULT_PURPOSE,
    DEFAULT_TARGET_ID,
    DEFAULT_TARGET_LABEL,
    MIN_PROMPT_CHARS,
    NUM_TESTS_PER_PLUGIN,
    PROMPTFOO_BIN,
    PROMPTFOO_TIMEOUT_SECONDS,
    REPORTS_DIR,
    TARGET_SAMPLES,
)


@dataclass
class PromptRecord:
    prompt: str
    probe: str
    attack_types: list[str]
    detector: str | None
    status: str | None
    is_hit: bool
    source_log: str


@dataclass
class PromptfooRunResult:
    config_paths: list[Path]
    tests_paths: list[Path]
    results_paths: list[Path]
    output_path: Path
    prompt_count: int


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()

def resolve_promptfoo_command() -> list[str]:
    """Return the base argv for invoking promptfoo (handles the Windows .cmd
    shim and an npx fallback)."""
    explicit = shutil.which(PROMPTFOO_BIN)
    if explicit:
        return [explicit]
    npx = shutil.which("npx")
    if npx:
        return [npx, PROMPTFOO_BIN]
    raise RuntimeError(
        "Could not find 'promptfoo' or 'npx' on PATH. "
        "Install it with: npm install -g promptfoo"
    )


def write_redteam_config(
    path: Path,
    target_id: str,
    target_label: str,
    purpose: str,
    plugins: list[str],
    strategies: list[str],
    num_tests: int,
    prompt_template: str = "{{prompt}}",
) -> None:
    config: dict[str, Any] = {
        "description": f"redteam: {', '.join(plugins)}",
        "targets": [{"id": target_id, "label": target_label}],
        "prompts": [prompt_template],
        "redteam": {
            "purpose": purpose,
            "numTests": num_tests,
            "plugins": [{"id": plugin} for plugin in plugins],
            "strategies": [{"id": strategy} for strategy in strategies],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _stream_subprocess(
    cmd: list[str],
    label: str,
    timeout_seconds: int,
    stop_event: threading.Event | None,
) -> list[str]:
    print(f"[Promptfoo] Starting ({label}): {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines: list[str] = []
    start_time = time.monotonic()

    def read_stream() -> None:
        if process.stdout:
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                try:
                    print(f"[Promptfoo] {line}", flush=True)
                except UnicodeEncodeError:
                    safe = line.encode("ascii", errors="replace").decode("ascii", errors="replace")
                    print(f"[Promptfoo] {safe}", flush=True)

    reader_thread = threading.Thread(target=read_stream, daemon=True)
    reader_thread.start()

    while process.poll() is None:
        if stop_event and stop_event.is_set():
            print(f"[Promptfoo] Received stop signal during {label}, terminating...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError("promptfoo run cancelled by user.")

        if timeout_seconds > 0 and (time.monotonic() - start_time) > timeout_seconds:
            print(f"[Promptfoo] Timeout exceeded ({timeout_seconds}s) during {label}, killing...")
            process.kill()
            for line in output_lines[-20:]:
                print(f"  {line}")
            raise RuntimeError(f"promptfoo {label} exceeded timeout of {timeout_seconds} seconds.")

        time.sleep(0.5)

    reader_thread.join(timeout=5)
    returncode = process.returncode or 0
    if returncode != 0:
        context = "\n".join(output_lines[-30:]) if output_lines else "(no output captured)"
        raise RuntimeError(f"promptfoo {label} failed with code {returncode}:\n{context}")

    return output_lines


def run_promptfoo_with_stop(
    target_id: str,
    target_label: str,
    purpose: str,
    plugins: list[str],
    strategies: list[str],
    num_tests: int,
    reports_dir: Path,
    timeout_seconds: int,
    run_tag: str,
    stop_event: threading.Event | None = None,
) -> tuple[Path, Path, Path]:
    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_path = reports_dir / f"{run_tag}.config.json"
    tests_path = reports_dir / f"{run_tag}.tests.yaml"
    results_path = reports_dir / f"{run_tag}.results.json"

    write_redteam_config(
        path=config_path,
        target_id=target_id,
        target_label=target_label,
        purpose=purpose,
        plugins=plugins,
        strategies=strategies,
        num_tests=num_tests,
    )

    base = resolve_promptfoo_command()

    generate_cmd = base + [
        "redteam", "generate",
        "-c", str(config_path),
        "-o", str(tests_path),
        "--no-progress-bar",
        "--force",
    ]
    _stream_subprocess(generate_cmd, f"generate:{run_tag}", timeout_seconds, stop_event)
    if not tests_path.exists():
        raise RuntimeError(f"promptfoo generate produced no test file at {tests_path}")

    eval_cmd = base + [
        "eval",
        "-c", str(tests_path),
        "-o", str(results_path),
        "--no-table",
        "--no-cache",
    ]
    _stream_subprocess(eval_cmd, f"eval:{run_tag}", timeout_seconds, stop_event)
    if not results_path.exists():
        raise RuntimeError(f"promptfoo eval produced no results file at {results_path}")

    return config_path, tests_path, results_path

def parse_results(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    results = data.get("results", {})
    if isinstance(results, dict):
        items = results.get("results", [])
        return [item for item in items if isinstance(item, dict)]
    return []


def extract_prompt_from_result(result: dict[str, Any]) -> str | None:
    prompt = result.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("raw"), str):
        raw = prompt["raw"]
        if raw.strip():
            return normalize_text(raw)

    variables = result.get("vars")
    if isinstance(variables, dict):
        candidate = variables.get("prompt")
        if isinstance(candidate, str) and candidate.strip():
            return normalize_text(candidate)
    return None


def is_hit_from_result(result: dict[str, Any]) -> bool:
    if result.get("error"):
        return False
    grading = result.get("gradingResult")
    if isinstance(grading, dict) and "pass" in grading:
        return not bool(grading["pass"])
    if "success" in result:
        return not bool(result["success"])
    return False


def detector_from_result(result: dict[str, Any]) -> str | None:
    test_case = result.get("testCase") or {}
    asserts = test_case.get("assert") or []
    detectors: list[str] = []
    for assertion in asserts:
        if isinstance(assertion, dict):
            value = assertion.get("type") or assertion.get("metric")
            if value:
                detectors.append(str(value))
    unique = sorted(set(detectors))
    return ", ".join(unique) if unique else None


def status_from_result(result: dict[str, Any]) -> str | None:
    grading = result.get("gradingResult")
    if isinstance(grading, dict):
        reason = grading.get("reason")
        if reason:
            return normalize_text(str(reason))[:200]
    return None


def records_from_results(
    results_path: Path,
    attack_type: str,
    plugin_attack_map: dict[str, list[str]],
    min_prompt_chars: int,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []

    for result in parse_results(results_path):
        if result.get("error"):
            continue

        prompt = extract_prompt_from_result(result)
        if not prompt or len(prompt) < min_prompt_chars:
            continue

        metadata = result.get("metadata") or {}
        plugin = str(metadata.get("pluginId") or "unknown")
        strategy = metadata.get("strategyId")
        probe = f"{plugin}/{strategy}" if strategy and strategy != "basic" else plugin

        records.append(
            PromptRecord(
                prompt=prompt,
                probe=probe,
                attack_types=sorted(plugin_attack_map.get(plugin, [attack_type])),
                detector=detector_from_result(result),
                status=status_from_result(result),
                is_hit=is_hit_from_result(result),
                source_log=results_path.name,
            )
        )

    return records


def dedupe_records(records: list[PromptRecord], max_prompts: int) -> list[PromptRecord]:
    unique: list[PromptRecord] = []
    seen: set[str] = set()

    for record in records:
        key = normalize_text(record.prompt).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(record)
        if len(unique) >= max_prompts:
            break

    return unique


def write_jsonl(path: Path, records: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

def build_dataset_from_promptfoo(
    target_id: str,
    target_label: str,
    purpose: str,
    attack_types: list[str],
    output_path: Path,
    reports_dir: Path,
    max_prompts: int,
    min_prompt_chars: int,
    num_tests: int,
    timeout_seconds: int,
    stop_event: threading.Event | None = None,
) -> PromptfooRunResult:
    invalid = sorted(set(attack_types) - set(ATTACK_TYPES.keys()))
    if invalid:
        raise ValueError(
            f"Unsupported attack_types: {', '.join(invalid)}. "
            f"Valid: {', '.join(sorted(ATTACK_TYPES.keys()))}"
        )

    plugin_attack_map = plugin_to_attack_type_map()
    all_records: list[PromptRecord] = []
    config_paths: list[Path] = []
    tests_paths: list[Path] = []
    results_paths: list[Path] = []

    for attack_type in attack_types:
        if stop_event and stop_event.is_set():
            raise RuntimeError("promptfoo run cancelled by user.")

        spec = ATTACK_TYPES[attack_type]
        plugins = spec.get("plugins", [])
        strategies = spec.get("strategies", []) or ["basic"]
        if not plugins:
            continue

        try:
            config_path, tests_path, results_path = run_promptfoo_with_stop(
                target_id=target_id,
                target_label=target_label,
                purpose=purpose,
                plugins=plugins,
                strategies=strategies,
                num_tests=num_tests,
                reports_dir=reports_dir,
                timeout_seconds=timeout_seconds,
                run_tag=attack_type,
                stop_event=stop_event,
            )
        except RuntimeError as exc:
            if stop_event and stop_event.is_set():
                raise
            print(f"[Promptfoo] Skipping attack_type '{attack_type}' after error: {exc}", flush=True)
            continue

        config_paths.append(config_path)
        tests_paths.append(tests_path)
        results_paths.append(results_path)

        all_records.extend(
            records_from_results(
                results_path=results_path,
                attack_type=attack_type,
                plugin_attack_map=plugin_attack_map,
                min_prompt_chars=min_prompt_chars,
            )
        )

    if not results_paths:
        raise RuntimeError("No promptfoo runs produced results.")

    unique_records = dedupe_records(records=all_records, max_prompts=max_prompts)
    write_jsonl(output_path, unique_records)

    return PromptfooRunResult(
        config_paths=config_paths,
        tests_paths=tests_paths,
        results_paths=results_paths,
        output_path=output_path,
        prompt_count=len(unique_records),
    )


def run_with_config_defaults(stop_event: threading.Event | None = None) -> PromptfooRunResult:
    return build_dataset_from_promptfoo(
        target_id=DEFAULT_TARGET_ID,
        target_label=DEFAULT_TARGET_LABEL,
        purpose=DEFAULT_PURPOSE,
        attack_types=sorted(ATTACK_TYPES.keys()),
        output_path=Path(DATASET_PATH),
        reports_dir=Path(REPORTS_DIR),
        max_prompts=TARGET_SAMPLES,
        min_prompt_chars=MIN_PROMPT_CHARS,
        num_tests=NUM_TESTS_PER_PLUGIN,
        timeout_seconds=PROMPTFOO_TIMEOUT_SECONDS,
        stop_event=stop_event,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run promptfoo red-teaming and build an attack-prompt dataset from its results."
    )
    parser.add_argument("--target-id", default=DEFAULT_TARGET_ID)
    parser.add_argument("--target-label", default=DEFAULT_TARGET_LABEL)
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument(
        "--attack-types",
        nargs="+",
        default=sorted(ATTACK_TYPES.keys()),
        help=f"Choose from: {', '.join(sorted(ATTACK_TYPES.keys()))}",
    )
    parser.add_argument("--output", default=DATASET_PATH)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    parser.add_argument("--max-prompts", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--num-tests", type=int, default=NUM_TESTS_PER_PLUGIN)
    parser.add_argument("--min-prompt-chars", type=int, default=MIN_PROMPT_CHARS)
    parser.add_argument("--timeout-seconds", type=int, default=PROMPTFOO_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = build_dataset_from_promptfoo(
        target_id=args.target_id,
        target_label=args.target_label,
        purpose=args.purpose,
        attack_types=args.attack_types,
        output_path=Path(args.output),
        reports_dir=Path(args.reports_dir),
        max_prompts=args.max_prompts,
        min_prompt_chars=args.min_prompt_chars,
        num_tests=args.num_tests,
        timeout_seconds=args.timeout_seconds,
    )

    for results_path in result.results_paths:
        print(f"promptfoo results log: {results_path}")
    print(f"Saved {result.prompt_count} prompts to {result.output_path}")


if __name__ == "__main__":
    main()
