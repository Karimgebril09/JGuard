import csv
import io
import json
from pathlib import Path


_RUNS: list[dict[str, object]] = [
    {
        "run_id": "run-001",
        "timestamp": "2026-06-01T10:00:00Z",
        "target_model": "qwen3:14b-q4_K_M",
        "strategy": "tool_based",
        "defenses_active": "obfuscation,multi_turn,roleplay",
        "success_rate": 0.21,
        "vulnerabilities": 4,
        "duration": "00:06:12",
        "critical": 1,
        "high": 1,
        "medium": 1,
        "low": 1,
    },
    {
        "run_id": "run-002",
        "timestamp": "2026-06-04T14:30:00Z",
        "target_model": "qwen3:14b-q4_K_M",
        "strategy": "custom_atj",
        "defenses_active": "obfuscation,roleplay",
        "success_rate": 0.32,
        "vulnerabilities": 7,
        "duration": "00:08:40",
        "critical": 2,
        "high": 2,
        "medium": 2,
        "low": 1,
    },
]

_RUNS_STORE_PATH = Path("outputs") / "eval_runs.json"


def _load_runs() -> list[dict[str, object]]:
    if _RUNS_STORE_PATH.exists():
        try:
            data = json.loads(_RUNS_STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                valid = [item for item in data if isinstance(item, dict)]
                if valid:
                    return valid
        except json.JSONDecodeError:
            pass
    return list(_RUNS)


def _save_runs(runs: list[dict[str, object]]) -> None:
    _RUNS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUNS_STORE_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def add_run(run: dict[str, object]) -> None:
    runs = _load_runs()
    runs.append(run)
    _save_runs(runs)


def get_summary() -> dict[str, object]:
    runs = _load_runs()
    if not runs:
        return {
            "total_campaigns": 0,
            "avg_jailbreak_success_rate": 0.0,
            "critical_issues_found": 0,
            "defense_blocked_sweeps_pct": 0.0,
        }

    total_campaigns = len(runs)
    avg_success = sum(float(run["success_rate"]) for run in runs) / total_campaigns
    critical_issues = sum(int(run["critical"]) for run in runs)
    blocked_pct = round(1 - avg_success, 4)
    return {
        "total_campaigns": total_campaigns,
        "avg_jailbreak_success_rate": round(avg_success, 4),
        "critical_issues_found": critical_issues,
        "defense_blocked_sweeps_pct": blocked_pct,
    }


def get_vulnerability_breakdown() -> dict[str, int]:
    runs = _load_runs()
    return {
        "critical": sum(int(run["critical"]) for run in runs),
        "high": sum(int(run["high"]) for run in runs),
        "medium": sum(int(run["medium"]) for run in runs),
        "low": sum(int(run["low"]) for run in runs),
    }


def get_attack_trends() -> list[dict[str, object]]:
    runs = _load_runs()
    return [
        {
            "run_id": str(run["run_id"]),
            "success_rate": float(run["success_rate"]),
        }
        for run in runs
    ]


def get_runs() -> list[dict[str, object]]:
    runs = _load_runs()
    fields = [
        "run_id",
        "timestamp",
        "target_model",
        "strategy",
        "defenses_active",
        "success_rate",
        "vulnerabilities",
        "duration",
    ]
    return [{field: run[field] for field in fields} for run in runs]


def compare_runs(baseline_run_id: str, compare_run_id: str) -> dict[str, object]:
    runs = _load_runs()
    baseline = next((run for run in runs if run["run_id"] == baseline_run_id), None)
    compare = next((run for run in runs if run["run_id"] == compare_run_id), None)
    if baseline is None or compare is None:
        raise KeyError("One or both run IDs were not found.")

    base_success = float(baseline["success_rate"])
    compare_success = float(compare["success_rate"])
    base_critical = int(baseline["critical"])
    compare_critical = int(compare["critical"])
    base_total = int(baseline["vulnerabilities"])
    compare_total = int(compare["vulnerabilities"])

    return {
        "jailbreak_success_rate": {
            "base": base_success,
            "compare": compare_success,
            "delta": round(compare_success - base_success, 4),
        },
        "critical_vulnerabilities": {
            "base": base_critical,
            "compare": compare_critical,
            "delta": compare_critical - base_critical,
        },
        "total_vulnerabilities": {
            "base": base_total,
            "compare": compare_total,
            "delta": compare_total - base_total,
        },
        "assessment_duration": {
            "base": str(baseline["duration"]),
            "compare": str(compare["duration"]),
        },
    }


def export_runs_as_json() -> bytes:
    return json.dumps(get_runs(), indent=2).encode("utf-8")


def export_runs_as_csv() -> bytes:
    runs = get_runs()
    if not runs:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(runs[0].keys()))
    writer.writeheader()
    writer.writerows(runs)
    return output.getvalue().encode("utf-8")
