from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .attacks import ATTACK_TYPES
from .config import (
    DATASET_PATH,
    DEFAULT_TARGET_NAME,
    DEFAULT_TARGET_TYPE,
    GARAK_MODULE,
    GARAK_TIMEOUT_SECONDS,
    MIN_PROMPT_CHARS,
    REPORTS_DIR,
    TARGET_SAMPLES,
)

PROMPT_KEYS = {
    "prompt",
    "prompts",
    "prompt_text",
    "probe_prompt",
    "user_prompt",
    "input",
    "question",
    "query",
    "content",
}


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
class GarakRunResult:
    report_log: Path
    hit_log: Path | None
    output_path: Path
    prompt_count: int


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def probe_to_attack_type_map() -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for attack_type, probes in ATTACK_TYPES.items():
        for probe in probes:
            reverse.setdefault(probe, []).append(attack_type)
    return reverse


def get_selected_probes(selected_attack_types: list[str]) -> list[str]:
    selected: list[str] = []
    for attack_type in selected_attack_types:
        selected.extend(ATTACK_TYPES.get(attack_type, []))
    return sorted(set(selected))


def run_garak(
    target_type: str,
    target_name: str,
    probes: list[str],
    reports_dir: Path,
    timeout_seconds: int,
) -> tuple[Path, Path | None]:
    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing_logs = {p.resolve(): p.stat().st_mtime_ns for p in reports_dir.glob("*.jsonl")}

    cmd = [
        sys.executable,
        "-m",
        GARAK_MODULE,
        "--target_type",
        target_type,
        "--target_name",
        target_name,
        "--probes",
        ",".join(probes),
        "--report_prefix",
        str(reports_dir / "run"),
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds if timeout_seconds > 0 else None,
        check=False,
    )

    if completed.returncode != 0:
        error_tail = completed.stderr[-2000:] if completed.stderr else completed.stdout[-2000:]
        raise RuntimeError(f"garak run failed with code {completed.returncode}:\n{error_tail}")

    new_logs: list[Path] = []
    for p in reports_dir.glob("*.jsonl"):
        resolved = p.resolve()
        mtime_ns = p.stat().st_mtime_ns
        previous_mtime = existing_logs.get(resolved)
        if previous_mtime is None or mtime_ns != previous_mtime:
            new_logs.append(resolved)

    if not new_logs:
        raise RuntimeError("No new GARAK JSONL logs were generated.")

    sorted_logs = sorted(new_logs, key=lambda p: p.stat().st_mtime, reverse=True)
    hit_log = next((p for p in sorted_logs if "hit" in p.name.lower()), None)
    report_log = next((p for p in sorted_logs if "report" in p.name.lower()), None)

    if report_log is None:
        report_log = sorted_logs[0]

    return report_log, hit_log


def run_garak_with_stop(
    target_type: str,
    target_name: str,
    probes: list[str],
    reports_dir: Path,
    timeout_seconds: int,
    stop_event: threading.Event | None = None,
) -> tuple[Path, Path | None]:
    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing_logs = {p.resolve(): p.stat().st_mtime_ns for p in reports_dir.glob("*.jsonl")}

    cmd = [
        sys.executable,
        "-m",
        GARAK_MODULE,
        "--target_type",
        target_type,
        "--target_name",
        target_name,
        "--probes",
        ",".join(probes),
        "--report_prefix",
        str(reports_dir / "run"),
    ]

    print(f"[Garak] Starting: {' '.join(cmd)}")
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
        """Stream process output to console and buffer."""
        if process.stdout:
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                try:
                    print(f"[Garak] {line}", flush=True)
                except UnicodeEncodeError:
                    safe_line = line.encode("ascii", errors="replace").decode("ascii", errors="replace")
                    print(f"[Garak] {safe_line}", flush=True)

    reader_thread = threading.Thread(target=read_stream, daemon=True)
    reader_thread.start()

    while process.poll() is None:
        if stop_event and stop_event.is_set():
            print("[Garak] Received stop signal, terminating...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError("GARAK run cancelled by user.")

        if timeout_seconds > 0 and (time.monotonic() - start_time) > timeout_seconds:
            print(f"[Garak] Timeout exceeded ({timeout_seconds}s), killing process...")
            process.kill()
            print("[Garak] Last output:")
            for line in output_lines[-20:]:
                print(f"  {line}")
            raise RuntimeError(f"GARAK run exceeded timeout of {timeout_seconds} seconds.")

        time.sleep(0.5)

    reader_thread.join(timeout=5)
    returncode = process.returncode or 0
    
    if returncode != 0:
        print(f"[Garak] Process exited with code {returncode}")
        error_context = "\n".join(output_lines[-30:]) if output_lines else "(no output captured)"
        raise RuntimeError(f"garak run failed with code {returncode}:\n{error_context}")

    new_logs: list[Path] = []
    for p in reports_dir.glob("*.jsonl"):
        resolved = p.resolve()
        mtime_ns = p.stat().st_mtime_ns
        previous_mtime = existing_logs.get(resolved)
        if previous_mtime is None or mtime_ns != previous_mtime:
            new_logs.append(resolved)

    if not new_logs:
        raise RuntimeError("No new GARAK JSONL logs were generated.")

    sorted_logs = sorted(new_logs, key=lambda p: p.stat().st_mtime, reverse=True)
    hit_log = next((p for p in sorted_logs if "hit" in p.name.lower()), None)
    report_log = next((p for p in sorted_logs if "report" in p.name.lower()), None)

    if report_log is None:
        report_log = sorted_logs[0]

    return report_log, hit_log


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def extract_prompt_candidates(data: Any, parent_key: str | None = None) -> list[str]:
    prompts: list[str] = []

    if isinstance(data, dict):
        if "messages" in data and isinstance(data["messages"], list):
            for message in data["messages"]:
                if isinstance(message, dict):
                    role = str(message.get("role", "")).lower()
                    content = message.get("content")
                    if isinstance(content, str) and role in {"user", "human", ""}:
                        prompts.append(content)

        for key, value in data.items():
            key_l = str(key).lower()
            if isinstance(value, str) and key_l in PROMPT_KEYS:
                prompts.append(value)
            else:
                prompts.extend(extract_prompt_candidates(value, key_l))

    elif isinstance(data, list):
        for value in data:
            prompts.extend(extract_prompt_candidates(value, parent_key))

    return prompts


def is_evaluated_entry(entry: dict[str, Any], require_hit: bool = True) -> bool:
    # Report log entries
    if entry.get("entry_type") == "attempt":
        if "status" not in entry:
            return False
        if require_hit and not entry.get("detector_results"):
            return False
        return True

    # Hitlog entries (no entry_type field)
    if "entry_type" not in entry:
        if "probe" in entry and "detector" in entry and "score" in entry:
            return True

    return False

def extract_prompt_from_entry(entry: dict[str, Any]) -> list[str]:
    prompts: list[str] = []

    # Report log: prompt is in conversations[*].turns[*].content.text (user role only)
    for convo in entry.get("conversations", []):
        for turn in convo.get("turns", []):
            if turn.get("role") == "user":
                text = turn.get("content", {}).get("text", "")
                if text:
                    prompts.append(normalize_text(text))
        break  # first conversation is enough, others are repetitions

    # Hitlog: prompt is in prompt.turns[*].content.text
    if not prompts:
        for turn in entry.get("prompt", {}).get("turns", []):
            if turn.get("role") == "user":
                text = turn.get("content", {}).get("text", "")
                if text:
                    prompts.append(normalize_text(text))

    return prompts


def is_hit_from_entry(entry: dict[str, Any]) -> bool:
    # Hitlog: has explicit score
    if "score" in entry:
        return float(entry["score"]) > 0.0

    # Report log: derive from detector_results
    for detector, scores in entry.get("detector_results", {}).items():
        if isinstance(scores, list) and any(s > 0.0 for s in scores):
            return True
        if isinstance(scores, (int, float)) and scores > 0.0:
            return True
    return False


def records_from_logs(
    report_log: Path,
    hit_log: Path | None,
    probe_attack_map: dict[str, list[str]],
    min_prompt_chars: int,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []

    # --- Report log ---
    for entry in parse_jsonl(report_log):
        if not is_evaluated_entry(entry, require_hit=False):
            continue

        probe = str(entry.get("probe_classname", entry.get("probe", "unknown")))
        detector = str(list(entry.get("detector_results", {}).keys())) or None
        status = entry.get("status")
        is_hit = is_hit_from_entry(entry)

        prompts = [p for p in extract_prompt_from_entry(entry) if len(p) >= min_prompt_chars]
        for prompt in prompts:
            records.append(PromptRecord(
                prompt=prompt,
                probe=probe,
                attack_types=sorted(probe_attack_map.get(probe, ["unknown"])),
                detector=detector,
                status=str(status) if status is not None else None,
                is_hit=is_hit,
                source_log=report_log.name,
            ))

    # --- Hit log ---
    if hit_log and hit_log.exists():
        for entry in parse_jsonl(hit_log):
            if not is_evaluated_entry(entry, require_hit=False):
                continue

            probe = str(entry.get("probe", "unknown"))
            detector = entry.get("detector")
            status = None  # hitlog has no status field
            prompts = [p for p in extract_prompt_from_entry(entry) if len(p) >= min_prompt_chars]

            for prompt in prompts:
                records.append(PromptRecord(
                    prompt=prompt,
                    probe=probe,
                    attack_types=sorted(probe_attack_map.get(probe, ["unknown"])),
                    detector=str(detector) if detector else None,
                    status=status,
                    is_hit=True,  # everything in hitlog is a hit by definition
                    source_log=hit_log.name,
                ))

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


def build_dataset_from_garak(
    target_type: str,
    target_name: str,
    attack_types: list[str],
    output_path: Path,
    reports_dir: Path,
    max_prompts: int,
    min_prompt_chars: int,
    timeout_seconds: int,
    stop_event: threading.Event | None = None,
) -> GarakRunResult:
    invalid = sorted(set(attack_types) - set(ATTACK_TYPES.keys()))
    if invalid:
        raise ValueError(
            f"Unsupported attack_types: {', '.join(invalid)}. Valid: {', '.join(sorted(ATTACK_TYPES.keys()))}"
        )

    probes = get_selected_probes(attack_types)
    if not probes:
        raise ValueError("No probes resolved from selected attack types.")

    probe_attack_map = probe_to_attack_type_map()
    report_log, hit_log = run_garak_with_stop(
        target_type=target_type,
        target_name=target_name,
        probes=probes,
        reports_dir=reports_dir,
        timeout_seconds=timeout_seconds,
        stop_event=stop_event,
    )

    records = records_from_logs(
        report_log=report_log,
        hit_log=hit_log,
        probe_attack_map=probe_attack_map,
        min_prompt_chars=min_prompt_chars,
    )
    unique_records = dedupe_records(records=records, max_prompts=max_prompts)
    write_jsonl(output_path, unique_records)

    return GarakRunResult(
        report_log=report_log,
        hit_log=hit_log,
        output_path=output_path,
        prompt_count=len(unique_records),
    )


def run_with_config_defaults(stop_event: threading.Event | None = None) -> GarakRunResult:
    """Run GARAK pipeline using default config"""
    return build_dataset_from_garak(
        target_type=DEFAULT_TARGET_TYPE,
        target_name=DEFAULT_TARGET_NAME,
        attack_types=sorted(ATTACK_TYPES.keys()),
        output_path=Path(DATASET_PATH),
        reports_dir=Path(REPORTS_DIR),
        max_prompts=TARGET_SAMPLES,
        min_prompt_chars=MIN_PROMPT_CHARS,
        timeout_seconds=GARAK_TIMEOUT_SECONDS,
        stop_event=stop_event,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GARAK and build an attack-prompt dataset from GARAK logs."
    )
    parser.add_argument("--target-type", default=DEFAULT_TARGET_TYPE)
    parser.add_argument("--target-name", default=DEFAULT_TARGET_NAME)
    parser.add_argument(
        "--attack-types",
        nargs="+",
        default=sorted(ATTACK_TYPES.keys()),
        help=f"Choose from: {', '.join(sorted(ATTACK_TYPES.keys()))}",
    )
    parser.add_argument("--output", default=DATASET_PATH)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    parser.add_argument("--max-prompts", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--min-prompt-chars", type=int, default=MIN_PROMPT_CHARS)
    parser.add_argument("--timeout-seconds", type=int, default=GARAK_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = build_dataset_from_garak(
        target_type=args.target_type,
        target_name=args.target_name,
        attack_types=args.attack_types,
        output_path=Path(args.output),
        reports_dir=Path(args.reports_dir),
        max_prompts=args.max_prompts,
        min_prompt_chars=args.min_prompt_chars,
        timeout_seconds=args.timeout_seconds,
    )

    print(f"GARAK report log: {result.report_log}")
    if result.hit_log:
        print(f"GARAK hit log: {result.hit_log}")
    print(f"Saved {result.prompt_count} prompts to {result.output_path}")


if __name__ == "__main__":
    main()
