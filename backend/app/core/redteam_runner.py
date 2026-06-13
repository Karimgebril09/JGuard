from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Literal, TypedDict, cast
from uuid import uuid4

from evaluation.redteaming_tools.garak.config import DEFAULT_TARGET_NAME, DEFAULT_TARGET_TYPE
from evaluation.redteaming_tools.garak.garak_pipeline import run_with_config_defaults

from backend.app.core import eval_store
from backend.app.core.garak_eval_metrics import build_garak_eval_record
from backend.app.models.redteam_models import LaunchCampaignRequest


class CampaignState(TypedDict):
    status: str
    progress_pct: int
    last_poll: int
    mode: Literal["garak", "mock"]
    stop_event: Event
    worker: Thread | None
    log_lines: list[str]


_CAMPAIGNS: dict[str, CampaignState] = {}
_CAMPAIGNS_LOCK = Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(state: CampaignState, message: str) -> None:
    cast(list[str], state["log_lines"]).append(f"[{_utcnow_iso()}] {message}")


def _run_garak_campaign(campaign_id: str, payload: LaunchCampaignRequest) -> None:
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        stop_event = campaign["stop_event"]

    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is not None:
            campaign["progress_pct"] = 20
            _append_log(campaign, "Starting Garak run with config defaults")
            _append_log(campaign, f"Target: {DEFAULT_TARGET_TYPE}/{DEFAULT_TARGET_NAME}")

    try:
        result = run_with_config_defaults(stop_event=stop_event)
    except Exception as exc:  # noqa: BLE001
        is_cancelled = stop_event.is_set() or "cancel" in str(exc).lower()
        new_status = "stopped" if is_cancelled else "failed"
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is None:
                return
            campaign["status"] = new_status
            campaign["progress_pct"] = 100
            if is_cancelled:
                _append_log(campaign, "Garak campaign stopped by user.")
            else:
                _append_log(campaign, f"Garak campaign failed: {exc}")
        return

    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        campaign["status"] = "completed"
        campaign["progress_pct"] = 100
        _append_log(campaign, f"Garak report log: {result.report_log}")
        if result.hit_log:
            _append_log(campaign, f"Garak hit log: {result.hit_log}")
        _append_log(campaign, f"Saved {result.prompt_count} prompts to {result.output_path}")

    try:
        eval_record = build_garak_eval_record(
            run_id=campaign_id,
            report_log=result.report_log,
            hit_log=result.hit_log,
            target_model=f"{DEFAULT_TARGET_TYPE}/{DEFAULT_TARGET_NAME}",
            strategy="tool_based",
            defenses_active="garak_default",
        )
        eval_store.add_run(eval_record)
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is not None:
                _append_log(campaign, f"Saved evaluation metrics for run_id={campaign_id}")
    except Exception as exc:  # noqa: BLE001
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is not None:
                _append_log(campaign, f"Metrics generation failed: {exc}")


def _start_garak_worker(campaign_id: str, payload: LaunchCampaignRequest) -> None:
    worker = Thread(
        target=_run_garak_campaign,
        args=(campaign_id, payload),
        daemon=True,
        name=f"garak-campaign-{campaign_id}",
    )
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        campaign["worker"] = worker
    worker.start()


def launch_campaign(payload: LaunchCampaignRequest) -> dict[str, object]:
    campaign_id = str(uuid4())
    now = _utcnow_iso()

    strategy = payload.strategy
    tool_framework = payload.tool_framework
    is_garak_campaign = strategy == "tool_based" and tool_framework == "garak"

    with _CAMPAIGNS_LOCK:
        _CAMPAIGNS[campaign_id] = CampaignState(
            status="running",
            progress_pct=5,
            last_poll=0,
            mode="garak" if is_garak_campaign else "mock",
            stop_event=Event(),
            worker=None,
            log_lines=[
                f"[{now}] Campaign initialized",
                f"Strategy: {strategy}",
                f"Tool framework: {tool_framework or 'none'}",
            ],
        )

        campaign = _CAMPAIGNS[campaign_id]
        if is_garak_campaign:
            _append_log(campaign, "Dispatching Garak runner")
        else:
            _append_log(campaign, "Using mock runner (real execution only enabled for tool_based + garak)")

    if is_garak_campaign:
        _start_garak_worker(campaign_id, payload)

    return {
        "campaign_id": campaign_id,
        "status": "started",
        "timestamp": now,
    }


def get_campaign_status(campaign_id: str) -> dict[str, object]:
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            raise KeyError(f"Unknown campaign_id '{campaign_id}'.")

        if campaign["status"] == "running" and campaign["mode"] == "mock":
            campaign["progress_pct"] = min(campaign["progress_pct"] + 15, 100)
            _append_log(campaign, f"Progress {campaign['progress_pct']}%")
            if campaign["progress_pct"] >= 100:
                campaign["status"] = "completed"
                _append_log(campaign, "Campaign completed")

        log_lines = campaign["log_lines"]
        last_poll = campaign["last_poll"]
        new_lines = log_lines[last_poll:]
        campaign["last_poll"] = len(log_lines)

        return {
            "campaign_id": campaign_id,
            "status": campaign["status"],
            "log_lines": new_lines,
            "progress_pct": campaign["progress_pct"],
        }


def stop_campaign(campaign_id: str) -> dict[str, object]:
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            raise KeyError(f"Unknown campaign_id '{campaign_id}'.")

        stop_event = campaign["stop_event"]
        stop_event.set()
        campaign["status"] = "stopped"
        campaign["progress_pct"] = max(campaign["progress_pct"], 100)
        _append_log(campaign, "Campaign stopped by user")

    return {
        "success": True,
        "campaign_id": campaign_id,
        "status": "stopped",
    }
